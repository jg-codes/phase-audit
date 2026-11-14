# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""
Multi-grid chunk selection on top of ChunkKV (phase-audit code).

ChunkKV scores fixed, non-overlapping chunks of ``chunk_length`` tokens (grid offset 0) and keeps the top chunks.
``TwoGridChunkKVPress`` scores the chunks of several grids (grid offset ``o`` starts chunks at positions
``o, o + L, o + 2L, ...`` plus a leading partial chunk ``[0, o)`` and a trailing partial chunk) with the same scorer
and the same chunk statistic as ChunkKV, takes chunks greedily by score across all grids, counts only tokens that are
not yet kept, and stops at exactly the number of tokens the ChunkKV selection of grid 0 keeps in that layer.

Everything not stated here is ChunkKV's: head-summed token scores, one index set shared by all KV heads of a layer,
chunk score = mean over the chunk's tokens of the head-summed score, the last partial chunk is an ordinary chunk.

Rules fixed for the greedy fill (phase-audit pre-registration C2G):
  * candidates are ordered by chunk score (descending), ties by grid rank (order of ``grid_offsets``), then start;
  * a candidate with ``n_new`` unkept tokens is added whole if ``n_new <= remaining budget``;
  * otherwise the ``remaining`` unkept tokens of that chunk with the highest head-summed token score are added
    (ties: lower position) and the fill stops;
  * a candidate with ``n_new == 0`` is skipped;
  * candidate scores are NOT recomputed after tokens are kept (stale-score rule, the registered primary rule): a chunk that
    overlaps an already kept chunk keeps its original mean score and is added for its not-yet-kept tokens;
  * ``lazy=True`` (exploratory arm) recomputes a candidate's score as the mean over its not-yet-kept tokens when it is
    popped and re-inserts it if the score dropped (lazy greedy), everything else unchanged;
  * ``select_chunk_length`` (exploratory arm) sets the candidate chunk length of the grids (default ``chunk_length``);
    the budget K is always the ChunkKV budget of ``chunk_length``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from kvpress.presses.chunkkv_press import ChunkKVPress


class _Clock:
    """Wall clock that works on CPU and CUDA without synchronising inside the item (events are read at item end)."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self.device = device

    def _event(self):
        with torch.cuda.device(self.device):
            e = torch.cuda.Event(enable_timing=True)
            e.record(torch.cuda.current_stream(self.device))
        return e

    def start(self):
        if self.cuda:
            return self._event()
        return time.perf_counter()

    def stop(self, t0):
        if self.cuda:
            return (t0, self._event())
        return time.perf_counter() - t0

    @staticmethod
    def resolve(x) -> float:
        if isinstance(x, tuple):
            x[1].synchronize()
            return x[0].elapsed_time(x[1]) / 1000.0
        return float(x)


def _timed_score(inner_score, log_list, clock_holder):
    assert not getattr(inner_score, "_c2g_wrapped", False), "inner press score already wrapped (shared inner press across arms)"

    def score(module, hidden_states, keys, values, attentions, kwargs):
        clock = clock_holder[0] or _Clock(keys.device)
        clock_holder[0] = clock
        t0 = clock.start()
        out = inner_score(module, hidden_states, keys, values, attentions, kwargs)
        log_list.append(clock.stop(t0))
        return out

    score._c2g_wrapped = True
    return score


def stock_chunk_selection(global_scores: torch.Tensor, kv_len: int, chunk_length: int, compression_ratio: float):
    """Verbatim ChunkKV (kvpress 0.5.x) chunk ranking. Returns (sorted kept indices (1D, cpu-device agnostic), tok_sum).

    ``global_scores`` has shape (1, n_kv_heads, kv_len). Requires kv_len >= chunk_length.
    """
    num_complete_chunks = kv_len // chunk_length
    remaining_tokens = kv_len % chunk_length
    main_scores = global_scores[..., : num_complete_chunks * chunk_length]
    main_chunk_scores = main_scores.sum(dim=1).view(-1, num_complete_chunks, chunk_length)
    main_chunk_scores = main_chunk_scores.mean(dim=-1)
    if remaining_tokens > 0:
        remaining_scores = global_scores[..., -remaining_tokens:]
        remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
        chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
    else:
        chunk_scores = main_chunk_scores
    n_chunks_kept = max(1, int((num_complete_chunks + (remaining_tokens > 0)) * (1 - compression_ratio)))
    top_chunks = chunk_scores.topk(n_chunks_kept, dim=-1)
    indices = []
    for chunk_idx in top_chunks.indices[0]:
        if chunk_idx < num_complete_chunks:
            start_idx = chunk_idx * chunk_length
            chunk_indices = torch.arange(start_idx, start_idx + chunk_length, device=global_scores.device)
        else:
            chunk_indices = torch.arange(num_complete_chunks * chunk_length, kv_len, device=global_scores.device)
        indices.append(chunk_indices)
    return torch.cat(indices).sort()[0]


def grid_chunks(tok: torch.Tensor, kv_len: int, chunk_length: int, offset: int):
    """Chunk table of one grid. ``tok`` = head-summed token scores, shape (1, kv_len), model dtype.

    Returns (starts, ends, scores) as python-int lists and a float32 numpy array. Chunk statistic and dtype handling
    mirror ChunkKV's (mean over the chunk of the head-summed score, computed in the score dtype).
    """
    starts, ends, scores = [], [], []
    if offset > 0:
        starts.append(0)
        ends.append(offset)
        scores.append(tok[:, :offset].mean(dim=-1).float())
    nc = (kv_len - offset) // chunk_length
    rem = (kv_len - offset) % chunk_length
    if nc > 0:
        main = tok[:, offset : offset + nc * chunk_length].reshape(1, nc, chunk_length).mean(dim=-1).float()
        for k in range(nc):
            starts.append(offset + k * chunk_length)
            ends.append(offset + (k + 1) * chunk_length)
        scores.append(main[0])
    if rem > 0:
        starts.append(offset + nc * chunk_length)
        ends.append(kv_len)
        scores.append(tok[:, -rem:].mean(dim=-1).float())
    sc = torch.cat([s.reshape(-1) for s in scores]).detach().cpu().numpy()
    assert len(starts) == len(sc) == len(ends)
    return starts, ends, sc


def greedy_fill(tok_cpu: np.ndarray, cands, budget: int, kv_len: int, lazy: bool = False):
    """Greedy chunk fill to exactly ``budget`` tokens. ``cands`` = list of (score, grid_rank, start, end).

    Returns (sorted kept indices, info) with info = {"fill_tokens": tokens added by the final partial-chunk rule}.
    """
    import heapq

    kept = np.zeros(kv_len, dtype=bool)
    remaining = budget
    fill_tokens = 0
    if not lazy:
        order = sorted(cands, key=lambda c: (-c[0], c[1], c[2]))
        stream = iter(order)
        pop = lambda: next(stream, None)  # noqa: E731
    else:
        heap = [(-c[0], c[1], c[2], c[3]) for c in cands]
        heapq.heapify(heap)

        def pop():
            while heap:
                ns, rank, s, e = heapq.heappop(heap)
                free = ~kept[s:e]
                nf = int(free.sum())
                if nf == 0:
                    continue
                true = float(tok_cpu[s:e][free].mean())
                if heap and true < -heap[0][0] - 1e-12:
                    heapq.heappush(heap, (-true, rank, s, e))
                    continue
                return (true, rank, s, e)
            return None

    while remaining > 0:
        c = pop()
        if c is None:
            break
        _, _, s, e = c
        free = np.nonzero(~kept[s:e])[0] + s
        n_new = len(free)
        if n_new == 0:
            continue
        if n_new <= remaining:
            kept[free] = True
            remaining -= n_new
        else:
            ts = tok_cpu[free]
            take = free[np.lexsort((free, -ts))[:remaining]]
            kept[take] = True
            fill_tokens = int(remaining)
            remaining = 0
    assert remaining == 0 and int(kept.sum()) == budget, (remaining, int(kept.sum()), budget)
    return np.nonzero(kept)[0], {"fill_tokens": fill_tokens}


@dataclass
class TwoGridChunkKVPress(ChunkKVPress):
    """
    Multi-grid ChunkKV.

    Parameters
    ----------
    grid_offsets : tuple[int, ...]
        Offsets of the chunk grids, each in [0, chunk_length). (0, 10) is the registered two-grid arm.
    layer_alternation : bool
        If True, ``grid_offsets`` must hold two offsets; even layers use the first, odd layers the second, one grid
        per layer (Swin-style alternation).
    force_greedy : bool
        Use the greedy path even for offsets (0,) (unit tests). Default False: offsets (0,) delegate to ChunkKV.
    lazy : bool
        Lazy greedy: re-score candidates on their not-yet-kept tokens when popped (exploratory arm).
    select_chunk_length : int or None
        Candidate chunk length of the grids (exploratory fine-chunk arm); budget stays that of ``chunk_length``.
    diagnostics : bool
        Compute overlap and fragmentation statistics after the timed section (off the clock).
    """

    grid_offsets: tuple = (0, 10)
    layer_alternation: bool = False
    force_greedy: bool = False
    lazy: bool = False
    select_chunk_length: int = None
    diagnostics: bool = True
    log: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self._sel_len = self.select_chunk_length or self.chunk_length
        assert all(0 <= o < self._sel_len for o in self.grid_offsets), self.grid_offsets
        assert len(set(self.grid_offsets)) == len(self.grid_offsets)
        if self.layer_alternation:
            assert len(self.grid_offsets) == 2
        self._clock = [None]
        self._t_score: list = []
        self._t_select: list = []
        self.press.score = _timed_score(self.press.score, self._t_score, self._clock)

    def reset_log(self):
        self.log = []
        self._t_score.clear()
        self._t_select.clear()

    def resolve_timings(self) -> dict:
        return {
            "score_s": float(sum(_Clock.resolve(x) for x in self._t_score)),
            "select_s": float(sum(_Clock.resolve(x) for x in self._t_select)),
        }

    def compress(self, module: nn.Module, hidden_states, keys, values, attentions, kwargs):
        if self.press.compression_ratio == 0:
            return keys, values
        assert attentions is None, "TwoGridChunkKVPress does not support attentions."
        assert keys.shape[0] == 1, "batch size 1 only (ChunkKV selection is written for batch 1)"
        assert hasattr(module, "layer_idx"), "module has no layer_idx"
        kv_len = keys.shape[2]
        if kv_len < self.chunk_length:
            out = self.press.compress(module, hidden_states, keys, values, attentions, kwargs)
            self.log.append({"layer": int(module.layer_idx), "kv_len": int(kv_len), "kept": int(out[0].shape[2]),
                             "chunkkv_kept": int(out[0].shape[2]), "overlap_with_chunkkv": int(out[0].shape[2]),
                             "offsets": [], "runs": None, "fill_tokens": 0})
            return out
        layer = int(module.layer_idx)
        clock = self._clock[0] or _Clock(keys.device)
        self._clock[0] = clock

        global_scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        t0 = clock.start()

        # budget and reference selection: ChunkKV's own (grid 0, chunk_length) for this layer
        ref = stock_chunk_selection(global_scores, kv_len, self.chunk_length, self.press.compression_ratio)
        budget = int(ref.numel())

        offsets = self.grid_offsets
        if self.layer_alternation:
            offsets = (offsets[layer % 2],)

        info = {"fill_tokens": 0}
        if offsets == (0,) and not self.force_greedy and self._sel_len == self.chunk_length and not self.lazy:
            idx = ref
        else:
            tok = global_scores.sum(dim=1)  # (1, kv_len), same op as ChunkKV
            cands = []
            for rank, o in enumerate(offsets):
                st, en, sc = grid_chunks(tok, kv_len, self._sel_len, o)
                cands.extend((float(sc[i]), rank, st[i], en[i]) for i in range(len(st)))
            tok_cpu = tok[0].float().detach().cpu().numpy()
            sel, info = greedy_fill(tok_cpu, cands, budget, kv_len, lazy=self.lazy)
            idx = torch.from_numpy(sel).to(keys.device)
        assert idx.numel() == budget
        indices = idx.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, keys.shape[-1])
        out_k = keys.gather(2, indices).contiguous()
        indices_v = idx.view(1, 1, -1, 1).expand(values.shape[0], values.shape[1], -1, values.shape[-1])
        out_v = values.gather(2, indices_v).contiguous()
        self._t_select.append(clock.stop(t0))
        entry = {"layer": layer, "kv_len": int(kv_len), "kept": int(budget), "chunkkv_kept": int(budget),
                 "offsets": [int(o) for o in offsets], "fill_tokens": int(info["fill_tokens"])}
        if self.diagnostics:
            a_ = idx.detach().cpu().numpy()
            entry["overlap_with_chunkkv"] = int(np.intersect1d(a_, ref.detach().cpu().numpy()).size)
            entry["runs"] = int(1 + (np.diff(a_) > 1).sum())
        self.log.append(entry)
        return out_k, out_v


@dataclass
class RecordingChunkKVPress(ChunkKVPress):
    """Stock ChunkKV (verbatim ``compress``) plus per-layer kept counts and score/selection timings."""

    log: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self._clock = [None]
        self._t_score: list = []
        self._t_total: list = []
        self.press.score = _timed_score(self.press.score, self._t_score, self._clock)

    def reset_log(self):
        self.log = []
        self._t_score.clear()
        self._t_total.clear()

    def resolve_timings(self) -> dict:
        score = sum(_Clock.resolve(x) for x in self._t_score)
        total = sum(_Clock.resolve(x) for x in self._t_total)
        return {"score_s": float(score), "select_s": float(total - score)}

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        assert hasattr(module, "layer_idx"), "module has no layer_idx"
        clock = self._clock[0] or _Clock(keys.device)
        self._clock[0] = clock
        t0 = clock.start()
        out_k, out_v = super().compress(module, hidden_states, keys, values, attentions, kwargs)
        self._t_total.append(clock.stop(t0))
        a_ = None
        self.log.append({"layer": int(module.layer_idx), "kv_len": int(keys.shape[2]),
                         "kept": int(out_k.shape[2]), "chunkkv_kept": int(out_k.shape[2]),
                         "overlap_with_chunkkv": int(out_k.shape[2]), "offsets": [0], "runs": None, "fill_tokens": 0})
        return out_k, out_v
