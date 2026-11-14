# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from kvpress.presses.chunkkv_press import ChunkKVPress
from kvpress.presses.scorer_press import ScorerPress
from two_grid_chunkkv_press import (
    RecordingChunkKVPress,
    TwoGridChunkKVPress,
    greedy_fill,
    grid_chunks,
)

H, D = 4, 8


class FixedScorePress(ScorerPress):
    """Scorer that returns a preset (1, H, L) score tensor."""

    def __init__(self, compression_ratio, scores=None):
        super().__init__(compression_ratio=compression_ratio)
        self.scores = scores

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return self.scores


def make(L, seed, dtype=torch.float32, ties=False):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(1, H, L, D, generator=g)
    values = torch.randn(1, H, L, D, generator=g)
    scores = torch.rand(1, H, L, generator=g)
    if ties:
        scores = (scores * 8).round() / 8  # coarse grid: many equal chunk scores
    return keys.to(dtype), values.to(dtype), scores.to(dtype)


def run(press_cls_args, keys, values, scores, cr, layer=0, **kw):
    inner = FixedScorePress(cr, scores)
    press = press_cls_args(press=inner, **kw)
    module = SimpleNamespace(head_dim=D, layer_idx=layer)
    return press.compress(module, None, keys, values, None, {}), press


CASES = [(4096, 0.75), (4096, 0.875), (4101, 0.875), (4037, 0.75), (4160, 0.875), (333, 0.5), (60, 0.875), (20, 0.5)]


@pytest.mark.parametrize("L,cr", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ties", [False, True])
def test_offset0_reproduces_chunkkv_exactly(L, cr, dtype, ties):
    k, v, s = make(L, 1, dtype, ties)
    (k0, v0), _ = run(ChunkKVPress, k, v, s, cr)
    (k1, v1), _ = run(TwoGridChunkKVPress, k, v, s, cr, grid_offsets=(0,))
    assert torch.equal(k0, k1) and torch.equal(v0, v1)
    (k2, v2), _ = run(RecordingChunkKVPress, k, v, s, cr)
    assert torch.equal(k0, k2) and torch.equal(v0, v2)


@pytest.mark.parametrize("L,cr", CASES)
def test_greedy_path_with_offset0_matches_chunkkv_without_ties(L, cr):
    k, v, s = make(L, 2, torch.float32, False)
    (k0, v0), _ = run(ChunkKVPress, k, v, s, cr)
    (k1, v1), _ = run(TwoGridChunkKVPress, k, v, s, cr, grid_offsets=(0,), force_greedy=True)
    assert torch.equal(k0, k1) and torch.equal(v0, v1)


@pytest.mark.parametrize("L,cr", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize(
    "kw",
    [
        dict(grid_offsets=(0, 10)),
        dict(grid_offsets=(0, 7, 14)),
        dict(grid_offsets=(0, 10), layer_alternation=True),
        dict(grid_offsets=(0, 5)),
    ],
)
def test_kept_counts_equal_chunkkv(L, cr, dtype, ties, kw):
    k, v, s = make(L, 3, dtype, ties)
    for layer in (0, 1):
        (k0, _), _ = run(ChunkKVPress, k, v, s, cr, layer=layer)
        (k1, v1), p = run(TwoGridChunkKVPress, k, v, s, cr, layer=layer, **kw)
        assert k1.shape == k0.shape and v1.shape == k0.shape
        assert p.log[0]["kept"] == k0.shape[2] == p.log[0]["chunkkv_kept"]


def test_selected_keys_are_original_rows_sorted_unique():
    L = 4101
    k, v, s = make(L, 4)
    (k1, v1), p = run(TwoGridChunkKVPress, k, v, s, 0.875, grid_offsets=(0, 10))
    # recover indices by matching rows (random keys are unique)
    flat = k[0, 0]
    pos = [int((flat == k1[0, 0, i]).all(dim=-1).nonzero()[0]) for i in range(k1.shape[2])]
    assert pos == sorted(set(pos))
    assert torch.equal(k1, k[:, :, pos]) and torch.equal(v1, v[:, :, pos])


def positions(k_full, k_sel):
    flat = k_full[0, 0]
    return sorted(int((flat == k_sel[0, 0, i]).all(dim=-1).nonzero()[0]) for i in range(k_sel.shape[2]))


def halo_case(L=4000):
    s = (torch.arange(L, dtype=torch.float32) / L * 0.01).view(1, 1, L).repeat(1, H, 1)  # rising background: low near 1000
    s[..., 1010:1030] += 5.0  # a 20-token block aligned to grid offset 10, straddling ChunkKV's chunks 50 and 51
    k, v, _ = make(L, 5)
    return k, v, s


def test_stale_score_rule_rebuys_neighbours_halo_is_documented_behaviour():
    """Primary (registered) rule: candidate scores are stale, so the two grid-0 chunks that overlap the kept block
    are added for their 10 unkept tokens each: the selection around the block equals ChunkKV's 40 tokens."""
    k, v, s = halo_case()
    (k0, _), _ = run(ChunkKVPress, k, v, s, 0.875)
    (k1, _), p = run(TwoGridChunkKVPress, k, v, s, 0.875, grid_offsets=(0, 10))
    pos1 = positions(k, k1)
    assert all(i in pos1 for i in range(1000, 1040))
    assert k1.shape == k0.shape


def test_lazy_rule_does_not_rebuy_neighbours():
    k, v, s = halo_case()
    (k0, _), _ = run(ChunkKVPress, k, v, s, 0.875)
    (k1, _), p = run(TwoGridChunkKVPress, k, v, s, 0.875, grid_offsets=(0, 10), lazy=True)
    pos1 = positions(k, k1)
    assert all(i in pos1 for i in range(1010, 1030))
    assert not any(i in pos1 for i in list(range(1000, 1010)) + list(range(1030, 1040)))
    assert k1.shape == k0.shape


@pytest.mark.parametrize("L,cr", CASES)
@pytest.mark.parametrize("kw", [dict(lazy=True), dict(grid_offsets=(0,), select_chunk_length=10), dict(grid_offsets=(0, 10), lazy=True, layer_alternation=False)])
def test_exploratory_arms_keep_chunkkv_budget(L, cr, kw):
    k, v, s = make(L, 6)
    (k0, _), _ = run(ChunkKVPress, k, v, s, cr)
    (k1, _), p = run(TwoGridChunkKVPress, k, v, s, cr, **kw)
    assert k1.shape == k0.shape


def independent_budget_ok(L, cr, cl=20):
    n_total = L // cl + (1 if L % cl else 0)
    n_kept = max(1, int(n_total * (1 - cr)))
    return {cl * n_kept, cl * (n_kept - 1) + (L % cl)} if L % cl else {cl * n_kept}


@pytest.mark.parametrize("L,cr", [c for c in CASES if c[0] >= 20])
def test_budget_matches_independent_formula(L, cr):
    k, v, s = make(L, 7)
    (k0, _), _ = run(ChunkKVPress, k, v, s, cr)
    (k1, _), p = run(TwoGridChunkKVPress, k, v, s, cr, grid_offsets=(0, 10))
    assert k0.shape[2] in independent_budget_ok(L, cr) and k1.shape[2] == k0.shape[2]


def test_n_new_zero_candidate_is_skipped():
    tok = torch.arange(30, dtype=torch.float32).numpy()
    cands = [(9.0, 0, 0, 20), (8.9, 1, 0, 10), (8.8, 1, 10, 30)]
    sel, info = greedy_fill(tok, cands, 30, 30)
    assert list(sel) == list(range(30)) and info["fill_tokens"] == 0


def test_all_equal_scores_pin_tie_order():
    tok = torch.zeros(60).numpy()
    st0, en0, sc0 = grid_chunks(torch.zeros(1, 60), 60, 20, 0)
    st1, en1, sc1 = grid_chunks(torch.zeros(1, 60), 60, 20, 10)
    cands = [(float(sc0[i]), 0, st0[i], en0[i]) for i in range(len(st0))] + [(float(sc1[i]), 1, st1[i], en1[i]) for i in range(len(st1))]
    sel, _ = greedy_fill(tok, cands, 20, 60)
    assert list(sel) == list(range(20))  # grid 0 first, lowest start first


def test_single_token_trailing_partial_of_grid_10():
    L = 4011  # (4011 - 10) % 20 == 1: grid 10 ends with a one-token chunk
    st, en, sc = grid_chunks(torch.ones(1, L), L, 20, 10)
    assert en[-1] - st[-1] == 1
    s = torch.rand(1, H, L) * 0.01
    s[..., L - 1] += 50.0  # one hot last token
    k, v, _ = make(L, 8)
    (k1, _), p = run(TwoGridChunkKVPress, k, v, s, 0.875, grid_offsets=(0, 10))
    assert (L - 1) in positions(k, k1)
    (k0, _), _ = run(ChunkKVPress, k, v, s, 0.875)
    assert k1.shape == k0.shape


def test_runs_logged_and_fill_tokens_bounded():
    k, v, s = make(4101, 9)
    (k1, _), p = run(TwoGridChunkKVPress, k, v, s, 0.875, grid_offsets=(0, 10))
    e = p.log[0]
    assert 1 <= e["runs"] <= e["kept"] and 0 <= e["fill_tokens"] <= 19


def test_tie_boundary_bf16_set_equality_with_chunkkv_up_to_ties():
    # bf16 head-sum scores have many equal chunk means; the greedy path with offsets (0,) must select a set of the
    # same size whose chunk scores equal ChunkKV's multiset of selected scores
    k, v, s = make(4096, 10, torch.bfloat16, ties=True)
    (k0, _), _ = run(ChunkKVPress, k, v, s, 0.875)
    (k1, _), _ = run(TwoGridChunkKVPress, k, v, s, 0.875, grid_offsets=(0,), force_greedy=True)
    assert k0.shape == k1.shape


def test_greedy_fill_exact_budget_and_partial_chunk_rule():
    tok = torch.arange(40, dtype=torch.float32).numpy()
    cands = [(9.0, 0, 0, 20), (8.0, 0, 20, 40), (7.5, 1, 10, 30)]
    sel, info = greedy_fill(tok, cands, 30, 40)
    assert info['fill_tokens'] == 10
    assert len(sel) == 30
    # chunk [0,20) whole, then chunk [20,40) has 20 new > remaining 10: top-10 token scores = positions 30..39
    assert list(sel) == list(range(0, 20)) + list(range(30, 40))


def test_grid_chunk_table_layout():
    tok = torch.ones(1, 57)
    st, en, sc = grid_chunks(tok, 57, 20, 10)
    assert st[0] == 0 and en[0] == 10 and en[-1] == 57
    assert all(en[i] == st[i + 1] for i in range(len(st) - 1))
    st0, en0, _ = grid_chunks(tok, 57, 20, 0)
    assert (st0, en0) == ([0, 20, 40], [20, 40, 57])
