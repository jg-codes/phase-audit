# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0

"""pa1 runner: chunk-phase pilot (query-aware protocol), Qwen3-0.6B, kvpress 0.5.5+c90, CPU.

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
ARMS = ["A1_chunkkv", "A1h_chunkkv_perhead", "A2_snapkv", "A3_twogrid", "A4_adaptive_layer",
        "A4h_adaptive_head", "A5_lowpass", "A6_layer_stagger"]


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


def cid_cached(n: int, o: int) -> torch.Tensor:
    if (n, o) not in _CID:
        _CID[(n, o)] = chunk_ids(n, o)
    return _CID[(n, o)]


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


def select_twogrid(ts: torch.Tensor, k: int, offsets=(0, 10)) -> torch.Tensor:
    n = ts.shape[0]
    cands = []
    for o in offsets:
        cid = cid_cached(n, o)
        means, _ = chunk_means(ts, cid)
        for j in range(means.shape[0]):
            cands.append((float(means[j]), o, j))
    cands.sort(key=lambda t: -t[0])
    cids = {o: cid_cached(n, o) for o in offsets}
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


def select_all(sc: torch.Tensor, layer: int, cr: float) -> tuple[dict[str, torch.Tensor], dict]:
    """sc: [H,N] SnapKV scores of one layer (window already set to max+1). Returns bool masks [H,N] per arm."""
    h, n = sc.shape
    ts = sc.sum(0)  # head-summed, as in kvpress ChunkKV
    m1 = select_kvpress_chunks(ts, cr)
    k = int(m1.sum().item())
    out: dict[str, torch.Tensor] = {}
    out["A1_chunkkv"] = m1.unsqueeze(0).expand(h, n)
    cid0 = cid_cached(n, 0)
    out["A1h_chunkkv_perhead"] = select_budget(sc, cid0, k)
    out["A2_snapkv"] = torch.stack([topk_mask(sc[i], k) for i in range(h)])
    out["A3_twogrid"] = select_twogrid(ts, k).unsqueeze(0).expand(h, n)
    m4, o4 = select_adaptive(ts.unsqueeze(0), k)
    out["A4_adaptive_layer"] = m4.expand(h, n)
    m4h, o4h = select_adaptive(sc, k)
    out["A4h_adaptive_head"] = m4h
    out["A5_lowpass"] = select_budget(tri_smooth(ts), cid0, k).unsqueeze(0).expand(h, n)
    o6 = (7 * layer) % C
    out["A6_layer_stagger"] = select_budget(ts, cid_cached(n, o6), k).unsqueeze(0).expand(h, n)
    info = {"k": k, "o4": int(o4[0]), "o4h": o4h.tolist(), "o6": o6}
    for a, m in out.items():
        expect(m.shape == (h, n) and m.dtype == torch.bool, f"{a} mask shape/dtype")
        expect(bool((m.sum(1) == k).all()), f"{a}: kept count != K={k} (layer {layer}): {m.sum(1).tolist()}")
    return out, info


# ------------------------------------------------------------------ recorder press
from kvpress.presses.base_press import BasePress  # noqa: E402
from kvpress.presses.chunkkv_press import ChunkKVPress  # noqa: E402
from kvpress.presses.snapkv_press import SnapKVPress  # noqa: E402


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
    ties: dict = field(default_factory=dict)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        li = module.layer_idx
        snap = SnapKVPress(compression_ratio=0.0, window_size=WINDOW, kernel_size=KERNEL)
        sc = snap.score(module, hidden_states, keys, values, attentions, kwargs)  # [1,H,N]
        expect(sc.dtype == torch.float32 and sc.shape[0] == 1 and sc.shape[2] == keys.shape[2], "score shape")
        sc2 = sc[0].detach()
        for cr in self.crs:
            ms, info = select_all(sc2, li, cr)
            for a, m in ms.items():
                self.masks.setdefault((cr, a), {})[li] = m.clone()
            self.infos.setdefault(cr, {})[li] = info
            if self.verify:
                real = ChunkKVPress(press=SnapKVPress(compression_ratio=cr, window_size=WINDOW, kernel_size=KERNEL),
                                    chunk_length=C)
                rk, _ = real.compress(module, hidden_states, keys, values, None, kwargs)
                idx1 = ms["A1_chunkkv"][0].nonzero().squeeze(1)
                mine = keys[:, :, idx1]
                ok1 = bool(torch.equal(sorted_rows(rk), sorted_rows(mine)))
                n = keys.shape[2]
                kk = info["k"]
                real2 = SnapKVPress(compression_ratio=1 - (kk + 0.5) / n, window_size=WINDOW, kernel_size=KERNEL)
                rk2, _ = real2.compress(module, hidden_states, keys, values, None, kwargs)
                expect(rk2.shape[2] == kk, f"real SnapKV kept {rk2.shape[2]} != {kk}")
                # real SnapKV vs mine: identical up to exact score ties at the K-th score (arbitrary tie-break)
                kth = sc2.topk(kk, dim=-1).values[:, -1:]
                strict = sc2 > kth
                ok2 = True
                for h in range(keys.shape[1]):
                    ok2 = ok2 and bool(torch.isin(keys[0, h, strict[h], 0], rk2[0, h, :, 0]).all())
                    ok2 = ok2 and bool(ms["A2_snapkv"][h][strict[h]].all())
                self.ties[(cr, li)] = int((sc2 == kth).sum().item())
                self.ver.setdefault(cr, []).append((ok1, ok2))
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


def make_items(sb: SynthBuilder, tok, n_items_essay: int, depths, seed: int):
    texts, f, col = load_essays(tok)
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    streams = []
    used = []
    j = 0
    while len(streams) < n_items_essay:
        buf: list[int] = []
        while len(buf) < sb.body + 50:
            buf += tok.encode("\n\n" + texts[order[j % len(order)]], add_special_tokens=False)
            used.append(order[j % len(order)])
            j += 1
        streams.append(buf)
    items = []
    P = len(sb.pre) + len(sb.instr)
    kidx = 0
    for si, E in enumerate(streams):
        for d in depths:
            keys = sb.keys[kidx:kidx + 4]
            kidx += 4
            vals = [str(rng.randrange(1000000, 10000000)) for _ in range(4)]
            # distractors: depths away from target
            cand = [x for x in (0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.92) if abs(x - d) > 0.1]
            dd = sorted(rng.sample(cand, 3))
            dist = [(keys[1 + i], vals[1 + i], P + int(dd[i] * sb.body) + rng.randrange(0, C)) for i in range(3)]
            items.append(dict(item=len(items), essay=si, depth=d, key=keys[0], val=vals[0], dist=dist, E=E))
    return items, dict(parquet=f, text_col=col, essay_ids=used)


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
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    expect(next(model.parameters()).dtype == torch.float32, "dtype")
    out = args.out
    os.makedirs(out, exist_ok=True)
    meta = dict(argv=sys.argv, mode=args.mode, crs=args.crs, arms=ARMS, chunk=C, window=WINDOW, kernel=KERNEL,
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
        rng.shuffle(jobs)
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
            info = dict(item=it["item"], depth=it["depth"], phase_needle=job["phase"], needle_start=pt)
            ts_ = pt
        N = len(ids)
        x = torch.tensor([ids])
        do_ver = (not args.noverify) and (n_done % args.verify_every == 0)
        rec = Recorder(crs=tuple(args.crs), verify=do_ver)
        from transformers import DynamicCache
        full = DynamicCache()
        tp = time.time()
        with torch.no_grad():
            with rec(model):
                model.model(input_ids=x, past_key_values=full)
        t_prefill = time.time() - tp
        expect(full.get_seq_length() == N, "full cache length")
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
                row = dict(ctx=ctx, mode=args.mode, cr=cr, arm=arm, correct=(int(val in ans) if decoded else -1), ans=ans[:60], N=N,
                           kept_per_layer=ks, kept_total=kept_total, eff_cr=1 - sum(ks) / (N * len(ks)),
                           val_start=vs, phase_val=vs % C, val_kept=retention(mb, v_span),
                           needle_kept=retention(mb, nd_span), ver_a1=(int(all(a for a, b in ver)) if ver else -1),
                           ver_a2=(int(all(b for a, b in ver)) if ver else -1), ties_a2=(sum(v for (cc, _), v in rec.ties.items() if cc == cr) if ver else -1), **info)
                if arm == "A4_adaptive_layer":
                    row["o4"] = [rec.infos[cr][li]["o4"] for li in range(len(ks))]
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
                       verify_ok=all(r.get("ver_a1", 1) != 0 and r.get("ver_a2", 1) != 0 for r in ctx_rows),
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
    # phase-shift property: a score bump in a chunk is kept by A1 wherever it sits inside the grid chunk
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="synth", choices=["synth", "ruler"])
    ap.add_argument("--out", default="out")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--n_ctx", type=int, default=3000)
    ap.add_argument("--n_essays", type=int, default=4)
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
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a)
