# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0

"""pa2 runner (extends pa1_runner.py): granularity vs grid-phase control for two-grid ChunkKV. Query-aware protocol, CPU.

One prefill per context records SnapKV token scores per layer (the compression hook of kvpress sees the
UNcompressed K/V at every layer, so one full prefill gives exactly the scores each press would see).
All arms/CRs are then selected from those scores, gathered into fresh DynamicCaches and decoded greedily.
Arm 1 and arm 2 are verified in-hook against the real kvpress ChunkKVPress / SnapKVPress.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, random, sys, time
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

C = 20  # chunk length
WINDOW, KERNEL = 64, 5
ARMS = ["A1_chunkkv", "A3_twogrid", "A7_chunk10", "A8_chunk5", "A9_twogrid10", "A10_threegrid20",
        "A11_token_shared", "A12_chunk20_off10", "A13_fourgrid20", "A14_twogrid20_0_5"]
# arm definitions: (chunk length, grid offsets); all select from ONE head-summed token score per layer (shared by heads), K tokens
ARM_GRIDS = {"A1_chunkkv": (20, (0,)), "A3_twogrid": (20, (0, 10)), "A7_chunk10": (10, (0,)), "A8_chunk5": (5, (0,)),
             "A9_twogrid10": (10, (0, 5)), "A10_threegrid20": (20, (0, 7, 14)), "A11_token_shared": (1, (0,)),
             "A12_chunk20_off10": (20, (10,)), "A13_fourgrid20": (20, (0, 5, 10, 15)), "A14_twogrid20_0_5": (20, (0, 5))}


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ------------------------------------------------------------------ selection (pure torch, 1-D scores)
def chunk_ids(n: int, o: int, c: int = C) -> torch.Tensor:
    """chunk id per token for a grid with boundaries at o + c*j (o in 0..c-1); o=0 reproduces kvpress ChunkKV
    (full chunks, then one short remainder chunk). o>0 gives a short head chunk [0,o)."""
    pos = torch.arange(n)
    if o == 0:
        return pos // c
    return torch.where(pos < o, torch.zeros_like(pos), 1 + (pos - o) // c)


def chunk_means(ts: torch.Tensor, cid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nch = int(cid.max().item()) + 1
    sums = torch.zeros(nch, dtype=ts.dtype).index_add_(0, cid, ts)
    sizes = torch.zeros(nch, dtype=torch.long).index_add_(0, cid, torch.ones_like(cid))
    return sums / sizes.to(ts.dtype), sizes


def select_kvpress_chunks(ts: torch.Tensor, cr: float) -> torch.Tensor:
    """Exactly kvpress ChunkKVPress token selection (offset 0) on a 1-D score vector (bool mask)."""
    n = ts.shape[0]
    cid = cid_cached(n, 0)
    means, sizes = chunk_means(ts, cid)
    nch = means.shape[0]
    n_keep = max(1, int(nch * (1 - cr)))
    top = means.topk(n_keep).indices
    keepc = torch.zeros(nch, dtype=torch.bool)
    keepc[top] = True
    return keepc[cid]


_CID: dict = {}


def cid_cached(n: int, o: int, c: int = C) -> torch.Tensor:
    if (n, o, c) not in _CID:
        _CID[(n, o, c)] = chunk_ids(n, o, c)
    return _CID[(n, o, c)]


def select_budget(ts: torch.Tensor, cid: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy whole chunks by mean score until budget k; the next chunk is filled with its best tokens.
    ts: [N] or [B,N]; returns bool of the same shape. Batched over rows (e.g. KV heads)."""
    one = ts.dim() == 1
    x = ts.unsqueeze(0) if one else ts
    b, n = x.shape
    nch = int(cid.max().item()) + 1
    sizes = torch.zeros(nch, dtype=torch.long).index_add_(0, cid, torch.ones_like(cid))
    means = torch.zeros(b, nch, dtype=x.dtype).index_add_(1, cid, x) / sizes.to(x.dtype)
    order = means.argsort(dim=1, descending=True, stable=True)
    cum = sizes[order].cumsum(1)
    n_whole = (cum <= k).sum(1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(nch).expand(b, nch).contiguous())
    mask = rank[:, cid] < n_whole.unsqueeze(1)
    used = torch.where(n_whole > 0, cum.gather(1, (n_whole - 1).clamp(min=0).unsqueeze(1)).squeeze(1), torch.zeros_like(n_whole))
    rest = k - used
    has = (rest > 0) & (n_whole < nch)
    if bool(has.any()):
        nxt = order.gather(1, n_whole.clamp(max=nch - 1).unsqueeze(1)).squeeze(1)
        inch = (cid.unsqueeze(0) == nxt.unsqueeze(1)) & has.unsqueeze(1)
        s = torch.where(inch, x, torch.full_like(x, float("-inf")))
        o2 = s.argsort(dim=1, descending=True, stable=True)
        r2 = torch.empty_like(o2)
        r2.scatter_(1, o2, torch.arange(n).expand(b, n).contiguous())
        mask = mask | (inch & (r2 < rest.unsqueeze(1)))
    return mask[0] if one else mask


def select_multigrid(ts: torch.Tensor, k: int, offsets: tuple, c: int = C) -> torch.Tensor:
    """Chunks of length c on grids at the given offsets; chunks taken greedily by mean score, overlaps counted once;
    the chunk that would overflow the budget is filled with its best not-yet-kept tokens. Exactly k tokens kept."""
    n = ts.shape[0]
    cands = []
    for o in offsets:
        cid = cid_cached(n, o, c)
        means, _ = chunk_means(ts, cid)
        for j in range(means.shape[0]):
            cands.append((float(means[j]), o, j))
    cands.sort(key=lambda t: -t[0])
    cids = {o: cid_cached(n, o, c) for o in offsets}
    mask = torch.zeros(n, dtype=torch.bool)
    kept = 0
    for _, o, j in cands:
        m = (cids[o] == j) & ~mask
        new = int(m.sum().item())
        if new == 0:
            continue
        if kept + new <= k:
            mask |= m
            kept += new
        else:
            idx = m.nonzero().squeeze(1)
            best = ts[idx].topk(k - kept).indices
            mask[idx[best]] = True
            kept = k
        if kept == k:
            break
    return mask


def select_adaptive(ts: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row, pick the grid offset in 0..C-1 maximising the total kept score. ts [B,N] -> masks [B,N], offsets [B]."""
    b, n = ts.shape
    best = torch.full((b,), -1e30, dtype=ts.dtype)
    best_o = torch.zeros(b, dtype=torch.long)
    best_m = torch.zeros(b, n, dtype=torch.bool)
    for o in range(C):
        m = select_budget(ts, cid_cached(n, o), k)
        tot = (ts * m).sum(1)
        better = tot > best
        best = torch.where(better, tot, best)
        best_o = torch.where(better, torch.full_like(best_o, o), best_o)
        best_m = torch.where(better.unsqueeze(1), m, best_m)
    return best_m, best_o


def tri_smooth(ts: torch.Tensor, width: int = C) -> torch.Tensor:
    h = width // 2
    w = torch.tensor([h - abs(i) for i in range(-h + 1, h)], dtype=ts.dtype)
    w = w / w.sum()
    x = F.pad(ts.view(1, 1, -1), (h - 1, h - 1), mode="replicate")
    return F.conv1d(x, w.view(1, 1, -1)).view(-1)


def topk_mask(ts: torch.Tensor, k: int) -> torch.Tensor:
    m = torch.zeros(ts.shape[0], dtype=torch.bool)
    m[ts.topk(k).indices] = True
    return m


def select_all(sc: torch.Tensor, layer: int, cr: float, ts_in: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], dict]:
    """sc: [H,N] SnapKV scores of one layer (window already set to max+1). Returns bool masks [H,N] per arm.
    Every arm selects from the head-summed score (as kvpress ChunkKV does) and shares the selection across heads."""
    h, n = sc.shape
    ts = sc.sum(0) if ts_in is None else ts_in  # ts_in: head sum taken in the model dtype (bf16 runs), as kvpress does
    expect(ts.shape == (n,) and ts.dtype == torch.float32, "ts shape/dtype")
    m1 = select_kvpress_chunks(ts, cr)
    k = int(m1.sum().item())
    out: dict[str, torch.Tensor] = {}
    for arm, (c_, offs) in ARM_GRIDS.items():
        if arm == "A1_chunkkv":
            m = m1
        elif len(offs) == 1:
            m = select_budget(ts, cid_cached(n, offs[0], c_), k)
        else:
            m = select_multigrid(ts, k, offs, c_)
        out[arm] = m.unsqueeze(0).expand(h, n)
    info = {"k": k}
    for a, m in out.items():
        expect(m.shape == (h, n) and m.dtype == torch.bool, f"{a} mask shape/dtype")
        expect(bool((m.sum(1) == k).all()), f"{a}: kept count != K={k} (layer {layer}): {m.sum(1).tolist()}")
    return out, info


# ------------------------------------------------------------------ recorder press
from kvpress.presses.base_press import BasePress  # noqa: E402
from kvpress.presses.chunkkv_press import ChunkKVPress  # noqa: E402
from kvpress.presses.snapkv_press import SnapKVPress  # noqa: E402


# ------------------------------------------------------------------ real kvpress press: multi-grid ChunkKV
from kvpress.presses.scorer_press import ScorerPress  # noqa: E402


@dataclass
class MultiGridChunkKVPress(BasePress):
    """ChunkKV on several chunk grids at once (kvpress press, independent of the offline selector in this file).

    Scores come from the wrapped ScorerPress (SnapKV) and are summed over KV heads, as in ChunkKVPress. Every grid has
    chunks of `chunk_length` tokens that start at `offset + j*chunk_length` (the first chunk of an offset grid is the short head
    [0, offset)). Chunks of all grids are ranked by mean score and taken greedily; tokens already kept are not counted again;
    the chunk that would overflow the budget is filled with its best not-yet-kept tokens. The budget is the number of tokens
    ChunkKVPress(chunk_length=budget_chunk_length) keeps at the same compression_ratio, so cost is equal by construction.
    offsets=(0,) with chunk_length=budget_chunk_length reproduces ChunkKVPress.
    """

    press: ScorerPress
    chunk_length: int = 20
    offsets: tuple = (0, 10)
    budget_chunk_length: int = 20
    last_indices: dict = field(default_factory=dict)

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "needs a ScorerPress"
        assert all(0 <= o < self.chunk_length for o in self.offsets), "offset outside [0, chunk_length)"

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.press.compression_ratio == 0:
            return keys, values
        assert attentions is None, "does not support attentions"
        n = keys.shape[2]
        bc = self.budget_chunk_length
        assert n % bc == 0, "budget rule implemented for kv_len divisible by budget_chunk_length"
        budget = max(1, int((n // bc) * (1 - self.press.compression_ratio))) * bc
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)  # [B, H, N]
        ts = scores.sum(dim=1)[0].float()  # [N]
        c = self.chunk_length
        cands = []  # (mean, grid, start, end) in append order: grid-major, chunk-ascending
        for g, o in enumerate(self.offsets):
            bounds = ([0] if o > 0 else []) + list(range(o, n, c)) + [n]
            for a, b in zip(bounds[:-1], bounds[1:]):
                cands.append((float(ts[a:b].mean()), g, a, b))
        order = sorted(range(len(cands)), key=lambda i: -cands[i][0])  # stable
        kept = torch.zeros(n, dtype=torch.bool)
        total = 0
        for i in order:
            _, _, a, b = cands[i]
            free = (~kept[a:b]).nonzero().squeeze(1) + a
            if free.numel() == 0:
                continue
            if total + free.numel() <= budget:
                kept[free] = True
                total += free.numel()
            else:
                best = ts[free].topk(budget - total).indices
                kept[free[best]] = True
                total = budget
            if total == budget:
                break
        idx = kept.nonzero().squeeze(1)
        assert idx.numel() == budget, f"kept {idx.numel()} != budget {budget}"
        self.last_indices[module.layer_idx] = idx.clone()
        gi = idx.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)
        return keys.gather(2, gi).contiguous(), values.gather(2, gi).contiguous()



def sorted_rows(x: torch.Tensor) -> torch.Tensor:
    order = x[..., 0].argsort(dim=-1)
    return x.gather(2, order.unsqueeze(-1).expand_as(x))


@dataclass
class Recorder(BasePress):
    crs: tuple = (0.75, 0.875)
    verify: bool = True
    masks: dict = field(default_factory=dict)   # (cr, arm) -> {layer: bool [H,N]}
    infos: dict = field(default_factory=dict)   # cr -> {layer: info}
    ver: dict = field(default_factory=dict)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        li = module.layer_idx
        snap = SnapKVPress(compression_ratio=0.0, window_size=WINDOW, kernel_size=KERNEL)
        sc = snap.score(module, hidden_states, keys, values, attentions, kwargs)  # [1,H,N]
        expect(sc.dtype in (torch.float32, torch.bfloat16) and sc.shape[0] == 1 and sc.shape[2] == keys.shape[2], "score shape/dtype")
        sc2 = sc[0].detach().to(torch.float32)
        ts_native = None if sc.dtype == torch.float32 else sc[0].detach().sum(0).to(torch.float32)
        for cr in self.crs:
            ms, info = select_all(sc2, li, cr, ts_native)
            for a, m in ms.items():
                self.masks.setdefault((cr, a), {})[li] = m.clone()
            self.infos.setdefault(cr, {})[li] = info
            if self.verify:
                n = keys.shape[2]
                kk = info["k"]
                oks = {}
                for arm, cl in (("A1_chunkkv", 20), ("A7_chunk10", 10), ("A8_chunk5", 5)):
                    nch = (n + cl - 1) // cl
                    expect(kk % cl == 0 and n % cl == 0, "verification needs K and N multiples of the chunk length")
                    # kvpress keeps int(nch * (1 - cr')) chunks; choose cr' so that this is exactly kk/cl chunks
                    real = ChunkKVPress(press=SnapKVPress(compression_ratio=1 - (kk / cl + 0.5) / nch, window_size=WINDOW,
                                                          kernel_size=KERNEL), chunk_length=cl)
                    rk, _ = real.compress(module, hidden_states, keys, values, None, kwargs)
                    expect(rk.shape[2] == kk, f"real ChunkKV({cl}) kept {rk.shape[2]} != {kk}")
                    idx1 = ms[arm][0].nonzero().squeeze(1)
                    oks[arm] = bool(torch.equal(sorted_rows(rk), sorted_rows(keys[:, :, idx1])))
                for arm in ("A3_twogrid", "A9_twogrid10", "A10_threegrid20", "A12_chunk20_off10", "A13_fourgrid20", "A14_twogrid20_0_5"):
                    c_, offs = ARM_GRIDS[arm]
                    rp = MultiGridChunkKVPress(press=SnapKVPress(compression_ratio=cr, window_size=WINDOW, kernel_size=KERNEL),
                                               chunk_length=c_, offsets=offs, budget_chunk_length=C)
                    rk, _ = rp.compress(module, hidden_states, keys, values, None, kwargs)
                    idx_press = rp.last_indices[li]
                    idx_mine = ms[arm][0].nonzero().squeeze(1)
                    oks[arm] = bool(torch.equal(idx_press, idx_mine)) and bool(torch.equal(rk, keys[:, :, idx_mine]))
                self.ver.setdefault(cr, []).append(oks)
        return keys, values


# ------------------------------------------------------------------ data
ADJ = "fair quiet brave rapid calm eager gentle jolly keen lively mighty noble proud quick sharp silent steady swift tidy vivid witty zesty amber bright clever dusty early fancy grand happy".split()
NOUN = "sprout meadow harbor lantern orchard pebble ribbon saddle thistle valley window anchor basket candle drawer engine forest garden hammer island jacket kettle ladder marble needle oyster pillow quarry rocket spindle tunnel".split()


def build_template(tok):
    sep = "#" * 37
    t = tok.apply_chat_template([{"role": "user", "content": "X" + sep}], add_generation_prompt=True, tokenize=False,
                                enable_thinking=False)
    pre, suf = t.split("X" + sep)
    return pre, suf


class SynthBuilder:
    def __init__(self, tok, n_ctx: int, seed: int):
        self.tok, self.n_ctx = tok, n_ctx
        pre, suf = build_template(tok)
        enc = lambda s: tok.encode(s, add_special_tokens=False)
        self.pre = enc(pre)
        self.instr = enc("A special magic number is hidden within the following text. Make sure to memorize it. "
                         "I will quiz you about the number afterwards.\n")
        self.suf_text = suf
        rng = random.Random(seed)
        combos = [f"{a}-{n}" for a in ADJ for n in NOUN]
        sig = {}
        for key in combos:
            nd = len(enc(f" One of the special magic numbers for {key} is: 1234567."))
            q = len(enc(f"\nWhat is the special magic number for {key} mentioned in the provided text?"))
            d = len(enc(suf + f"The special magic number for {key} mentioned in the provided text is"))
            sig.setdefault((nd, q, d), []).append(key)
        best = max(sig, key=lambda s: len(sig[s]))
        self.nd, self.qlen, self.dlen = best
        self.keys = sig[best]
        rng.shuffle(self.keys)
        # value-digit offset inside the needle
        ids = enc(" One of the special magic numbers for fair-sprout is: 1234567.")
        digs = [tok.encode(str(d), add_special_tokens=False)[0] for d in "1234567"]
        self.voff = next(i for i in range(len(ids)) if ids[i:i + 7] == digs)
        expect(len(self.keys) >= 20, f"too few keys {len(self.keys)}")
        self.body = n_ctx - len(self.pre) - len(self.instr) - self.qlen
        self.enc = enc

    def needle(self, key: str, val: str) -> list[int]:
        ids = self.enc(f" One of the special magic numbers for {key} is: {val}.")
        expect(len(ids) == self.nd, "needle length")
        return ids

    def make(self, E: list[int], key: str, val: str, depth: float, phase: int, dist: list[tuple[str, str, int]]):
        """Return ids [N], target needle start index, value start index. Filler is position-indexed from E."""
        N = self.n_ctx
        P = len(self.pre) + len(self.instr)
        ids = [0] * N
        ids[:len(self.pre)] = self.pre
        ids[len(self.pre):P] = self.instr
        for i in range(P, N - self.qlen):
            ids[i] = E[i - P]
        m = (P + int(depth * self.body)) // C
        pt = C * m + phase
        spans = []
        nd = self.needle(key, val)
        ids[pt:pt + self.nd] = nd
        spans.append((pt, pt + self.nd))
        for (dk, dv, dp) in dist:
            s, e = dp, dp + self.nd
            expect(all(e <= a or s >= b for a, b in spans), "needle overlap")
            expect(P <= s and e <= N - self.qlen, "needle outside body")
            ids[s:e] = self.needle(dk, dv)
            spans.append((s, e))
        expect(pt + self.nd <= N - self.qlen, "target outside body")
        q = self.enc(f"\nWhat is the special magic number for {key} mentioned in the provided text?")
        expect(len(q) == self.qlen, "question length")
        ids[N - self.qlen:] = q
        dec = self.enc(self.suf_text + f"The special magic number for {key} mentioned in the provided text is")
        expect(len(dec) == self.dlen, "decode length")
        return ids, pt, pt + self.voff, dec


def load_essays(tok):
    import pandas as pd
    f = glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", "datasets--alessiodevoto--paul_graham_essays/snapshots/*/data/*.parquet"))[0]
    d = pd.read_parquet(f)
    col = max((c for c in d.columns if d[c].dtype == object), key=lambda c: d[c].astype(str).str.len().mean())
    return d[col].astype(str).tolist(), f, col


def make_items(sb: SynthBuilder, tok, n_hay: int, depths, seed: int, stride: int = 60000):
    """The essay dataset is ONE row of about 700k tokens (pa1 review: all pa1 streams were the same window).
    Haystack h = the window that starts at the first paragraph break (token containing a double newline) at or after
    token h*stride; haystack 0 starts at token 0 and is identical to the pa1 haystack."""
    texts, f, col = load_essays(tok)
    assert len(texts) == 1, f"expected the one-row essay text, got {len(texts)} rows"
    full = tok.encode("\n\n" + texts[0], add_special_tokens=False)
    nl = np.array([("\n\n" in tok.decode([t])) for t in full[: n_hay * stride + 5000]])
    rng = random.Random(seed)
    streams, starts = [], []
    for h in range(n_hay):
        s0 = 0 if h == 0 else int(np.nonzero(nl[h * stride:])[0][0] + h * stride)
        starts.append(s0)
        buf = full[s0: s0 + sb.body + 50]
        expect(len(buf) == sb.body + 50, "haystack window too short")
        streams.append(buf)
    expect(len(set(starts)) == n_hay and all(b - a >= sb.body + 50 for a, b in zip(starts, starts[1:])), "haystack windows overlap")
    items = []
    P = len(sb.pre) + len(sb.instr)
    kidx = 0
    for si, E in enumerate(streams):
        for d in depths:
            keys = sb.keys[kidx:kidx + 4]
            kidx += 4
            vals = [str(rng.randrange(1000000, 10000000)) for _ in range(4)]
            cand = [x for x in (0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.92) if abs(x - d) > 0.1]
            dd = sorted(rng.sample(cand, 3))
            dist = [(keys[1 + i], vals[1 + i], P + int(dd[i] * sb.body) + rng.randrange(0, C)) for i in range(3)]
            items.append(dict(item=len(items), essay=si, depth=d, key=keys[0], val=vals[0], dist=dist, E=E))
    expect(len(sb.keys) >= 4 * len(items), f"not enough distinct keys: {len(sb.keys)} < {4 * len(items)}")
    return items, dict(parquet=f, text_col=col, hay_starts=starts, hay_stride=stride, n_text_tokens=len(full))


# ------------------------------------------------------------------ model
def decode(model, tok, cache, dec_ids: list[int], pos0: int, max_new: int) -> str:
    ids = torch.tensor([dec_ids])
    pos = torch.arange(pos0, pos0 + ids.shape[1]).unsqueeze(0)
    out = model(input_ids=ids, past_key_values=cache, position_ids=pos, logits_to_keep=1)
    nxt = out.logits[0, -1].argmax()
    gen = [nxt]
    stops = model.generation_config.eos_token_id
    stops = stops if isinstance(stops, list) else [stops]
    p = pos[:, -1:] + 1
    for i in range(max_new - 1):
        if gen[-1].item() in stops:
            break
        if len(gen) >= 7 and all(tok.decode(g.view(1)).strip().isdigit() and len(tok.decode(g.view(1)).strip()) == 1 for g in gen[-7:]):
            break
        out = model(input_ids=gen[-1].view(1, 1), past_key_values=cache, position_ids=p + i)
        gen.append(out.logits[0, -1].argmax())
    return tok.decode(torch.stack(gen), skip_special_tokens=True)


def build_cache(full, masks_by_layer: dict[int, torch.Tensor]):
    from transformers import DynamicCache
    c = DynamicCache()
    for li in range(len(full.layers)):
        m = masks_by_layer[li]
        k = full.layers[li].keys
        v = full.layers[li].values
        idx = torch.stack([m[h].nonzero().squeeze(1) for h in range(m.shape[0])])  # [H,K]
        gi = idx.unsqueeze(0).unsqueeze(-1).expand(1, -1, -1, k.shape[3])
        c.update(k.gather(2, gi).contiguous(), v.gather(2, gi).contiguous(), li)
    return c


def retention(masks_by_layer, span: tuple[int, int]) -> float:
    s, e = span
    return float(np.mean([m[:, s:e].float().mean().item() for m in masks_by_layer.values()]))


def run(args):
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tdt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=tdt, attn_implementation="sdpa")
    model.eval()
    expect(next(model.parameters()).dtype == tdt, "dtype")
    out = args.out
    os.makedirs(out, exist_ok=True)
    meta = dict(argv=sys.argv, model=args.model, dtype=args.dtype, grids=ARM_GRIDS, mode=args.mode, crs=args.crs, arms=ARMS, chunk=C, window=WINDOW, kernel=KERNEL,
                torch=torch.__version__, threads=args.threads)
    import kvpress, transformers
    meta.update(kvpress=getattr(kvpress, "__version__", "?"), transformers=transformers.__version__)

    jobs = []  # list of dict(ctx_id, ids, dec, meta)
    if args.mode == "synth":
        sb = SynthBuilder(tok, args.n_ctx, args.seed)
        items, prov = make_items(sb, tok, args.n_essays, args.depths, args.seed)
        meta.update(prov=prov, nd=sb.nd, voff=sb.voff, qlen=sb.qlen, dlen=sb.dlen, n_keys=len(sb.keys),
                    prefix_len=len(sb.pre), instr_len=len(sb.instr), body=sb.body)
        phases = args.phases
        for it in items:
            for ph in phases:
                jobs.append(dict(it=it, phase=ph))
        rng = random.Random(args.seed + 1)
        # item blocks in item order (any prefix of the run is a set of complete items); phases shuffled within an item
        by_item: dict = {}
        for jb in jobs:
            by_item.setdefault(jb["it"]["item"], []).append(jb)
        jobs = []
        for it_id in sorted(by_item):
            blk = by_item[it_id]
            rng.shuffle(blk)
            jobs += blk
        if args.items:
            jobs = [jb for jb in jobs if jb["it"]["item"] in set(args.items)]
    else:
        import pandas as pd
        f = glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", "datasets--simonjegou--ruler/snapshots/*/4096/test-00000-of-00001.parquet"))[0]
        d = pd.read_parquet(f)
        d = d[d.task == args.task].reset_index(drop=True)
        pick = list(range(len(d)))
        random.Random(args.seed).shuffle(pick)
        pre, suf = build_template(tok)
        for i in pick[:args.n_items]:
            r = d.iloc[i]
            ids = tok.encode(pre + r.context + r.question, add_special_tokens=False)
            dec = tok.encode(suf + r.answer_prefix, add_special_tokens=False)
            val = r.answer[0]
            digs = [tok.encode(x, add_special_tokens=False)[0] for x in val]
            hits = [j for j in range(len(ids) - 6) if ids[j:j + 7] == digs]
            if len(hits) != 1:
                continue
            jobs.append(dict(ruler=dict(row=int(i), ids=ids, dec=dec, val=val, vs=hits[0])))
        meta.update(parquet=f, task=args.task, n_jobs=len(jobs))
    json.dump(meta, open(f"{out}/meta.json", "w"), default=str)

    rows_path = f"{out}/rows.jsonl"
    done = set()
    if os.path.exists(rows_path):
        for ln in open(rows_path):
            r = json.loads(ln)
            done.add(r["ctx"])
    t0 = time.time()
    fh = open(rows_path, "a")
    n_total = len(jobs)
    n_done = len(done)
    t_ctx = []
    for ji, job in enumerate(jobs):
        if "ruler" in job:
            ctx = f"ruler{job['ruler']['row']}"
        else:
            ctx = f"i{job['it']['item']}_p{job['phase']}"
        if ctx in done:
            continue
        if os.path.exists(f"{out}/STOP") or (time.time() - t0) > args.deadline_s or (args.max_ctx and len(t_ctx) >= args.max_ctx):
            break
        tc = time.time()
        if "ruler" in job:
            R = job["ruler"]
            ids, dec, val, vs = R["ids"], R["dec"], R["val"], R["vs"]
            info = dict(item=R["row"], depth=None, phase_needle=None)
            ts_ = vs
        else:
            it = job["it"]
            ids, pt, vs, dec = sb.make(it["E"], it["key"], it["val"], it["depth"], job["phase"], it["dist"])
            val = it["val"]
            info = dict(item=it["item"], essay=it["essay"], depth=it["depth"], phase_needle=job["phase"], needle_start=pt)
            ts_ = pt
        N = len(ids)
        x = torch.tensor([ids])
        do_ver = (not args.noverify) and (len(t_ctx) % args.verify_every == 0)
        rec = Recorder(crs=tuple(args.crs), verify=do_ver)
        from transformers import DynamicCache
        full = DynamicCache()
        tp = time.time()
        with torch.no_grad():
            with rec(model):
                model.model(input_ids=x, past_key_values=full)
        t_prefill = time.time() - tp
        expect(full.get_seq_length() == N, "full cache length")
        if args.e2e and len(t_ctx) < args.e2e:
            # gate: the REAL MultiGridChunkKVPress, run through the kvpress prefill hook of the model, must leave exactly the cache
            # (keys and values, every layer and KV head) that the offline masks gather from the full cache
            e2e_rows = []
            for cr in args.crs:
                for arm in ("A3_twogrid", "A9_twogrid10", "A10_threegrid20", "A13_fourgrid20", "A14_twogrid20_0_5"):
                    c_, offs = ARM_GRIDS[arm]
                    pr = MultiGridChunkKVPress(press=SnapKVPress(compression_ratio=cr, window_size=WINDOW, kernel_size=KERNEL),
                                               chunk_length=c_, offsets=offs, budget_chunk_length=C)
                    pc = DynamicCache()
                    with torch.no_grad():
                        with pr(model):
                            model.model(input_ids=x, past_key_values=pc)
                    ref = build_cache(full, rec.masks[(cr, arm)])
                    same = all(torch.equal(pc.layers[li].keys, ref.layers[li].keys) and torch.equal(pc.layers[li].values, ref.layers[li].values)
                               for li in range(len(ref.layers)))
                    e2e_rows.append(dict(ctx=ctx, cr=cr, arm=arm, identical_cache=int(same), n_layers=len(ref.layers)))
                    del pc, ref
            with open(f"{out}/e2e.jsonl", "a") as fe:
                for r_ in e2e_rows:
                    fe.write(json.dumps(r_) + "\n")
            expect(all(r_["identical_cache"] == 1 for r_ in e2e_rows), f"real press != offline selector: {e2e_rows}")
        nd_span = (ts_, ts_ + 20 if "ruler" in job else ts_ + sb.nd)
        v_span = (vs, vs + 7)
        ctx_rows = []
        for cr in args.crs:
            ver = rec.ver.get(cr, [])
            ks = [rec.infos[cr][li]["k"] for li in range(len(full.layers))]
            for arm in ARMS:
                mb = rec.masks[(cr, arm)]
                for li, m in mb.items():
                    expect(int(m.sum(1).min()) == ks[li] == int(m.sum(1).max()), f"equal-cost fail {arm} L{li}")
                kept_total = int(sum(ks) * full.layers[0].keys.shape[1])
                cache = build_cache(full, mb)
                expect(cache.get_seq_length(0) == ks[0], "cache len")
                ans = ""
                decoded = (not args.nodecode) and (args.decode_arms is None or arm in args.decode_arms)
                if decoded:
                    with torch.no_grad():
                        ans = decode(model, tok, cache, dec, N, args.max_new)
                vkey = arm if arm in ("A1_chunkkv", "A7_chunk10", "A8_chunk5", "A3_twogrid", "A9_twogrid10", "A10_threegrid20", "A12_chunk20_off10", "A13_fourgrid20", "A14_twogrid20_0_5") else None
                verd = (int(all(v[vkey] for v in ver)) if (ver and vkey) else -1)
                sha = hashlib.sha1(b"".join(mb[li][0].numpy().tobytes() for li in sorted(mb))).hexdigest()[:16]
                row = dict(ctx=ctx, mode=args.mode, cr=cr, arm=arm, correct=(int(val in ans) if decoded else -1), ans=ans[:60], N=N,
                           kept_per_layer=ks, kept_total=kept_total, eff_cr=1 - sum(ks) / (N * len(ks)),
                           val_start=vs, phase_val=vs % C, val_kept=retention(mb, v_span),
                           needle_kept=retention(mb, nd_span), ver_real=verd, mask_sha1=sha, **info)
                ctx_rows.append(row)
                del cache
            # cross-arm equal-cost
            kt = {r["arm"]: r["kept_per_layer"] for r in ctx_rows if r["cr"] == cr}
            expect(all(v == kt["A1_chunkkv"] for v in kt.values()), "kept counts differ across arms")
        if not args.nodecode:
            with torch.no_grad():
                ans = decode(model, tok, full, dec, N, args.max_new)
        else:
            ans = ""
        ctx_rows.append(dict(ctx=ctx, mode=args.mode, cr=0.0, arm="full", correct=int(val in ans), ans=ans[:60], N=N,
                             val_start=vs, phase_val=vs % C, val_kept=1.0, needle_kept=1.0, **info))
        for r in ctx_rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        n_done += 1
        t_ctx.append(time.time() - tc)
        eta = (n_total - n_done) * float(np.mean(t_ctx[-10:]))
        json.dump(dict(done=n_done, total=n_total, elapsed_s=time.time() - t0, eta_s=eta, last_ctx=ctx,
                       t_prefill=t_prefill, s_per_ctx=float(np.mean(t_ctx[-10:])), rss_gb=_rss(),
                       verify_ok=all(r.get("ver_real", 1) != 0 for r in ctx_rows),
                       state="running"), open(f"{out}/status.json", "w"))
        del full, rec
    st = json.load(open(f"{out}/status.json")) if os.path.exists(f"{out}/status.json") else {}
    st["state"] = "finished" if n_done >= n_total else "partial_stop"
    st["done"], st["total"] = n_done, n_total
    json.dump(st, open(f"{out}/status.json", "w"))


def _rss() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def selftest():
    torch.manual_seed(0)
    h, n = 8, 3000
    sc = torch.rand(h, n)
    sc[:, -64:] = sc.max() + 1
    for li in (0, 5, 13):
        for cr in (0.75, 0.875):
            ms, info = select_all(sc, li, cr)
            ks = {a: set(m.sum(1).tolist()) for a, m in ms.items()}
            assert all(v == {info["k"]} for v in ks.values()), ks
            assert info["k"] == 20 * int(150 * (1 - cr)), info
    # arm1 equals kvpress chunk logic on a non-multiple length
    sc = torch.rand(h, 3013)
    sc[:, -64:] = sc.max() + 1
    ms, info = select_all(sc, 1, 0.75)
    assert info["k"] == 20 * (int(151 * .25) - 1) + 13, info
    # new arms: single-offset multigrid == budgeted single grid; token arm == top-k of the shared score; equal K everywhere
    torch.manual_seed(1)
    sc = torch.rand(h, 3000)
    sc[:, -64:] = sc.max() + 1
    ts = sc.sum(0)
    for cr in (0.75, 0.875):
        ms, info = select_all(sc, 3, cr)
        k = info["k"]
        for c_, o in ((20, 0), (10, 0), (5, 0), (20, 10)):
            a = select_multigrid(ts, k, (o,), c_)
            b = select_budget(ts, cid_cached(3000, o, c_), k)
            assert bool((a == b).all()), (c_, o)
        assert bool((ms["A11_token_shared"][0] == topk_mask(ts, k)).all())
        assert all(int(m.sum(1).min()) == k for m in ms.values())
        assert int(ms["A12_chunk20_off10"][0].sum()) == k
        # multigrid keeps a hot block regardless of its phase: a 20-token hot block at every phase, K at CR 0.875
        for ph in range(20):
            t2 = torch.rand(3000) * 0.01
            t2[1000 + ph:1020 + ph] += 1.0
            kept1 = int(select_kvpress_chunks(t2, cr)[1000 + ph:1020 + ph].sum())
            kept3 = int(select_multigrid(t2, k, (0, 10), 20)[1000 + ph:1020 + ph].sum())
            kept10 = int(select_budget(t2, cid_cached(3000, 0, 10), k)[1000 + ph:1020 + ph].sum())
            assert kept3 >= 10 and kept10 >= 10, (ph, kept1, kept3, kept10)
            assert kept10 == 20 if ph % 10 == 0 else kept10 >= 10
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="synth", choices=["synth", "ruler"])
    ap.add_argument("--out", default="out")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--n_ctx", type=int, default=3000)
    ap.add_argument("--n_essays", type=int, default=10, help="number of distinct haystack windows")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--phases", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--crs", type=float, nargs="+", default=[0.75, 0.875])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--deadline_s", type=float, default=3.5 * 3600)
    ap.add_argument("--max_ctx", type=int, default=0)
    ap.add_argument("--task", default="niah_multikey_1")
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--noverify", action="store_true")
    ap.add_argument("--verify_every", type=int, default=8)
    ap.add_argument("--nodecode", action="store_true")
    ap.add_argument("--decode_arms", nargs="+", default=None)  # added after the main synth run; default = all arms (unchanged behaviour)
    ap.add_argument("--e2e", type=int, default=0, help="run the real press end-to-end on the first N contexts")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--items", type=int, nargs="+", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a)
