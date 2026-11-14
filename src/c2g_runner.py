#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""C2G runner: ChunkKV reproduction and multi-grid ChunkKV on kvpress RULER-4096 (query-aware track).

One process = one (model, CR) cell, one or more arms, a list of dataset row ids. Rows go to
<out>/<run_id>_rows.jsonl (append, flushed per item), gates to <out>/<run_id>_gates.json.

Arms: chunkkv_repro (stock ChunkKV, leaderboard config), two_grid (offsets 0,10), swin (0 even layers / 10 odd layers),
three_grid (0,7,14). Everything else follows evaluation/evaluate.py of NVIDIA/kvpress: dataset simonjegou/ruler
data_dir 4096 split test, context + question concatenated (query_aware), question "", answer_prefix, max_new_tokens
from the dataset row, greedy decoding, scorer = evaluation/benchmarks/ruler/calculate_metrics.py per item.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time

T_START = time.time()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NP_PATTERN = re.compile(r"[\x00-\x1f]")
TASKS = ["cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multiquery", "niah_multivalue",
         "niah_single_1", "niah_single_2", "niah_single_3", "qa_1", "qa_2", "vt"]
ARMS = {
    "chunkkv_repro": None,
    "chunkkv_repro_b": None,  # second independent run of the stock ChunkKV arm (run-to-run noise bound, smoke only)
    "two_grid": dict(grid_offsets=(0, 10), layer_alternation=False),
    "swin": dict(grid_offsets=(0, 10), layer_alternation=True),
    "three_grid": dict(grid_offsets=(0, 7, 14), layer_alternation=False),
    "two_grid_lazy": dict(grid_offsets=(0, 10), layer_alternation=False, lazy=True),
    "fine10": dict(grid_offsets=(0,), layer_alternation=False, select_chunk_length=10),
}


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def item_score(task, pred, answers):
    """Per-item score exactly as calculate_metrics (without the final rounding), 0-100."""
    pred = NP_PATTERN.sub("", pred.strip()).strip().lower()
    refs = [r.lower() for r in answers]
    if task.split("_")[0] == "qa":
        return 100.0 * max(1.0 if r in pred else 0.0 for r in refs)
    return 100.0 * sum(1.0 if r in pred else 0.0 for r in refs) / len(refs)


RULER4096_SHA256 = "47951890333763a6eede43bc5dbdc9f4f028f312a312f7299db05f7485ad1444"


def budget_candidates(L, cr, cl=20):
    """Kept-token counts ChunkKV can produce for context length L (its int(n*(1-CR)) rule; the partial last chunk may be in)."""
    n_total = L // cl + (1 if L % cl else 0)
    n_kept = max(1, int(n_total * (1 - cr)))
    return {cl * n_kept, cl * (n_kept - 1) + (L % cl)} if L % cl else {cl * n_kept}


def read_rows(path, model, cr, items_set, tolerate_truncated=True):
    """Read rows of an earlier run; validate model/CR/item membership, tolerate one truncated last line."""
    out, good_bytes = {}, 0
    with open(path, "rb") as f:
        data = f.read()
    lines = data.split(b"\n")
    for k, line in enumerate(lines):
        if not line.strip():
            good_bytes += len(line) + 1
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            assert tolerate_truncated and k >= len(lines) - 2, f"corrupt line {k} in {path}"
            print(f"WARNING: truncated last line in {path} dropped", flush=True)
            break
        assert r["model"] == model and abs(r["cr_nominal"] - cr) < 1e-9, (path, r["model"], r["cr_nominal"])
        assert items_set is None or r["item_id"] in items_set, (path, r["item_id"])
        key = (r["arm"], r["item_id"])
        assert key not in out, f"duplicate row {key} in {path}"
        out[key] = r
        good_bytes += len(line) + 1
    return out, min(good_bytes, len(data))


def load_ruler(data_dir="4096"):
    """Dataset simonjegou/ruler, data_dir 4096, split test, read from the local HF cache parquet (offline)."""
    import glob

    pat = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub",
                       "datasets--simonjegou--ruler", "snapshots", "*", data_dir, "test-00000-of-00001.parquet")
    hits = sorted(glob.glob(pat))
    assert len(hits) == 1, (pat, hits)
    sha = sha256_file(hits[0])
    assert sha == RULER4096_SHA256, f"RULER-4096 parquet differs from the registered one: {sha}"
    df = pd.read_parquet(hits[0])
    assert len(df) == 6500, df.shape
    assert df["task"].value_counts().min() == 500 and df["task"].nunique() == 13
    for col in ("context", "question", "answer_prefix", "answer", "task", "max_new_tokens"):
        assert col in df.columns, col
    return df


def load_published(path, df):
    """Published per-item predictions; assert row alignment with the dataset (format safety)."""
    pub = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(pub) == len(df) == 6500
    assert sorted(pub.columns) == sorted(["answer_prefix", "answer", "task", "predicted_answer", "question", "max_new_tokens", "compression_ratio"]), list(pub.columns)
    assert (pub["task"].values == df["task"].values).all(), "task order differs from dataset"
    assert (pub["answer_prefix"].values == df["answer_prefix"].fillna("").values).all()
    assert (pub["max_new_tokens"].astype(int).values == df["max_new_tokens"].astype(int).values).all()
    ds_ans = [list(a) for a in df["answer"]]
    # predictions.csv stores str(numpy array); compare it with the dataset's answers in the same representation
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()  # noqa: E731
    bad = [i for i in range(len(df)) if norm(df["answer"].iat[i]) != norm(pub["answer"].iat[i])]
    assert not bad, f"answers differ from dataset on {len(bad)} rows, first {bad[:5]}"
    return pub, ds_ans


def nvsmi_thread(path, stop):
    def run():
        with open(path, "w") as f:
            f.write("t,util,mem_mib\n")
            while not stop.is_set():
                try:
                    o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10).stdout.strip().split(",")
                    f.write(f"{time.time():.0f},{o[0].strip()},{o[1].strip()}\n"); f.flush()
                except Exception as e:  # noqa: BLE001
                    f.write(f"{time.time():.0f},err,{e}\n"); f.flush()
                stop.wait(60)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def build_press(arm, cr):
    from kvpress import SnapKVPress
    from kvpress.presses.two_grid_chunkkv_press import RecordingChunkKVPress, TwoGridChunkKVPress

    inner = SnapKVPress(compression_ratio=cr, window_size=64, kernel_size=5)
    if arm.startswith("chunkkv_repro"):
        return RecordingChunkKVPress(press=inner, chunk_length=20)
    return TwoGridChunkKVPress(press=inner, chunk_length=20, **ARMS[arm])


def sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cr", type=float, required=True)
    ap.add_argument("--arms", required=True, help="comma list; chunkkv_repro first when G3 is asserted")
    ap.add_argument("--items", required=True, help="json file with a list of dataset row ids")
    ap.add_argument("--published", required=True, help="published ChunkKV predictions.csv for this model and CR")
    ap.add_argument("--published-metrics", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--resume", nargs="*", default=[], help="earlier rows jsonl files of THIS run id (same model, CR, arms)")
    ap.add_argument("--repro-rows", nargs="*", default=[], help="chunkkv_repro rows (G1/G2 cross-check where items overlap)")
    ap.add_argument("--payload-sha", default="", help="sha256 of the payload tarball (bound into the gate file)")
    ap.add_argument("--expect-dtype", default="bfloat16")
    ap.add_argument("--g3", action="store_true", help="assert G3 after the chunkkv_repro arm, before treatment arms")
    ap.add_argument("--g3-floor-pp", type=float, default=2.8)
    ap.add_argument("--g3-match", type=float, default=0.5)
    ap.add_argument("--require-g3", default=None, help="gate file that must show G3 pass for this model and CR")
    ap.add_argument("--first-item-deadline-s", type=float, default=300.0)
    ap.add_argument("--meta", default="{}")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    pre = os.path.join(a.out, a.run_id)
    arms = a.arms.split(",")
    assert all(x in ARMS for x in arms), arms
    items = json.load(open(a.items))
    items = items["item_ids"] if isinstance(items, dict) else items
    assert len(items) == len(set(items)) and all(type(i) is int and 0 <= i < 6500 for i in items)
    items_sha = sha256_file(a.items)
    if a.require_g3:
        g = json.load(open(a.require_g3))
        assert g.get("G3_pass") is True and g.get("model") == a.model and abs(g.get("cr") - a.cr) < 1e-9, g
        assert g.get("runner_sha256") == sha256_file(__file__), "G3 gate was issued for another runner version"
        assert g.get("press_sha256") == sha256_file(os.path.join(HERE, "kvpress", "presses", "two_grid_chunkkv_press.py")), "G3 gate was issued for another press version"
        assert not a.payload_sha or g.get("payload_sha256") == a.payload_sha, "G3 gate was issued for another payload"
    if a.g3:
        assert arms[0] == "chunkkv_repro"
    assert a.cr in (0.75, 0.875), a.cr
    dev = torch.device(a.device)

    from transformers import pipeline

    import kvpress  # noqa: F401
    from timebox import TimeBox

    df = load_ruler()
    pub, ds_ans = load_published(a.published, df)
    if a.published_metrics:
        m = json.load(open(a.published_metrics))
        for t in TASKS:
            sc = [item_score(t, p, ds_ans[i]) for i, p in enumerate(pub["predicted_answer"]) if df["task"].iat[i] == t]
            got = round(sum(sc) / len(sc), 2)
            assert abs(got - m[t]["string_match"]) < 0.011, (t, got, m[t])
    # query-aware: context + question, question ""  (evaluate.py)
    ctx = (df["context"] + df["question"]).tolist()
    ap_ = df["answer_prefix"].fillna("").tolist()
    # the published entry's own configuration must say the CR we think it says
    import yaml

    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(a.published), "config.yaml")))
    assert abs(float(cfg["compression_ratio"]) - a.cr) < 1e-9 and cfg["query_aware"] is True and cfg["press_name"] == "chunkkv", cfg
    assert "chunk_length=20" in cfg["press_init_command"] and "window_size=64" in cfg["press_init_command"] and "kernel_size=5" in cfg["press_init_command"]

    mk = {"dtype": a.dtype if a.dtype == "auto" else getattr(torch, a.dtype), "attn_implementation": a.attn}
    meta_sha = {"runner": sha256_file(__file__), "press": sha256_file(os.path.join(HERE, "kvpress", "presses", "two_grid_chunkkv_press.py"))}
    t_l0 = time.time()
    pipe = pipeline("kv-press-text-generation", model=a.model, model_kwargs=mk, device=a.device)
    pipe.model.eval()
    if a.device.startswith("cuda"):
        assert pipe.model.config._attn_implementation == a.attn, pipe.model.config._attn_implementation
    n_layers = pipe.model.config.num_hidden_layers
    if a.expect_dtype:
        assert str(pipe.model.dtype) == f"torch.{a.expect_dtype}", pipe.model.dtype
    assert os.path.abspath(kvpress.__file__).startswith(HERE), kvpress.__file__
    t_loaded = time.time()

    # time attribution hooks
    dec = {"t": 0.0}
    orig_gen = pipe.generate_answer

    def gen_timed(*args, **kw):
        sync(a.device); t0 = time.perf_counter()
        out = orig_gen(*args, **kw)
        sync(a.device); dec["t"] += time.perf_counter() - t0
        return out

    pipe.generate_answer = gen_timed

    stop = threading.Event()
    if a.device.startswith("cuda"):
        nvsmi_thread(pre + "_nvsmi.csv", stop)

    items_set = set(items)
    done = {}
    rows_path = pre + "_rows.jsonl"
    if os.path.exists(rows_path):
        assert os.path.abspath(rows_path) in [os.path.abspath(p) for p in a.resume], f"{rows_path} exists and is not in --resume"
    for p in a.resume:
        rr, good = read_rows(p, a.model, a.cr, items_set)
        for k, r in rr.items():
            assert r["run_id"] == a.run_id and k[0] in arms, (k, r["run_id"])
            assert k not in done, f"duplicate row {k} across resume files"
            assert r["runner_sha256"] == meta_sha["runner"] and r["press_sha256"] == meta_sha["press"], "resume rows from another code version"
            done[k] = r
        if os.path.abspath(p) == os.path.abspath(rows_path) and good < os.path.getsize(p):
            with open(p, "r+b") as f:
                f.truncate(good)
    fout = open(rows_path, "a")
    repro_ref = {}
    for p in a.repro_rows:
        rr, _ = read_rows(p, a.model, a.cr, None)
        repro_ref.update({k[1]: r for k, r in rr.items() if k[0] == "chunkkv_repro"})
    meta = dict(json.loads(a.meta))
    meta.update(dict(
        run_id=a.run_id, model=a.model, cr=a.cr, arms=arms, n_items=len(items), device=a.device, attn=a.attn,
        gpu=(torch.cuda.get_device_name(dev) if a.device.startswith("cuda") else "cpu"), torch=torch.__version__,
        kvpress_file=kvpress.__file__, runner_sha256=meta_sha['runner'], items_sha256=items_sha, payload_sha256=a.payload_sha,
        press_sha256=meta_sha['press'], model_dtype=str(pipe.model.dtype),
        container_start_to_model_loaded_s=t_loaded - T_START, model_load_s=t_loaded - t_l0, n_layers=n_layers,
    ))
    import transformers
    meta["transformers"] = transformers.__version__
    json.dump(meta, open(pre + "_meta.json", "w"), indent=1)

    from transformers import DynamicCache

    cnt = dict(n_rows=0, g1_budget_formula=0, g1_cache_len=0, g1_vs_repro=0, g2_vs_repro=0)

    def run_item(arm, press, i):
        press.reset_log()
        if a.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(dev)
        dec["t"] = 0.0
        cache = DynamicCache()
        sync(a.device); t0 = time.perf_counter()
        out = pipe(ctx[i], questions=[""], answer_prefix=ap_[i], press=press, cache=cache,
                   max_new_tokens=int(df["max_new_tokens"].iat[i]), max_context_length=None)
        sync(a.device); t_total = time.perf_counter() - t0
        pred = out["answers"][0]
        tm = press.resolve_timings()
        log = press.log
        assert len(log) == n_layers, (len(log), n_layers)
        L = log[0]["kv_len"]
        assert all(x["kv_len"] == L for x in log)
        assert [x["layer"] for x in log] == list(range(n_layers)), "layers out of order"
        kept = [x["kept"] for x in log]
        assert kept == [x["chunkkv_kept"] for x in log], "G1 (internal): kept counts differ from the press's own budget"
        # independent checks: ChunkKV's budget formula, and the cache the model actually decoded from
        assert all(k in budget_candidates(L, a.cr) for k in kept), f"G1: kept counts {set(kept)} not producible by ChunkKV at L={L}, CR={a.cr}"
        cnt["g1_budget_formula"] += 1
        cache_len = [cache.get_seq_length(l) for l in range(n_layers)]
        assert cache_len == kept, "G1: cache length after prefill differs from the logged kept counts"
        cnt["g1_cache_len"] += 1
        kept_total = int(sum(kept))
        task = df["task"].iat[i]
        row = dict(run_id=a.run_id, arm=arm, model=a.model, cr_nominal=a.cr, item_id=int(i), task=task,
                   runner_sha256=meta_sha["runner"], press_sha256=meta_sha["press"],
                   predicted_answer=pred, score=item_score(task, pred, ds_ans[i]), kv_len=int(L),
                   kept_per_layer=kept, kept_total=kept_total, effective_cr=1.0 - kept_total / (L * n_layers),
                   overlap_with_chunkkv=[x.get("overlap_with_chunkkv") for x in log], runs_per_layer=[x.get("runs") for x in log],
                   fill_tokens_per_layer=[x.get("fill_tokens") for x in log],
                   score_s=tm["score_s"], select_s=tm["select_s"], decode_s=dec["t"],
                   prefill_s=t_total - dec["t"] - tm["score_s"] - tm["select_s"], total_s=t_total,
                   peak_mem_gb=(torch.cuda.max_memory_allocated(dev) / 2**30 if a.device.startswith("cuda") else None),
                   pub_pred=pub["predicted_answer"].iat[i], pub_score=item_score(task, pub["predicted_answer"].iat[i], ds_ans[i]))
        return row

    code_ids = dict(runner_sha256=meta_sha["runner"], press_sha256=meta_sha["press"], payload_sha256=a.payload_sha, items_sha256=items_sha)
    gates = dict(run_id=a.run_id, model=a.model, cr=a.cr, arms=arms, G3="not asserted", G3_pass=None,
                 first_item_started_after_container_start_s=None, **code_ids)
    tb = TimeBox(total=len(items) * len(arms), time_limit_s=a.time_limit, out_dir=pre + "_tb", log=lambda s: print(s, flush=True))
    rows_by_arm = {x: {} for x in arms}
    for k, r in done.items():
        if k[0] in rows_by_arm:
            rows_by_arm[k[0]][k[1]] = r
    presses = {x: build_press(x, a.cr) for x in arms}
    expired = False
    first = True
    for arm in arms:
        press = presses[arm]
        for i in items:
            if (arm, i) in done:
                continue
            if tb.expired():
                expired = True
                break
            if first:
                elapsed = time.time() - T_START
                gates["first_item_started_after_container_start_s"] = elapsed
                if elapsed > a.first_item_deadline_s:
                    print(f"ABORT: first item not started within {a.first_item_deadline_s}s ({elapsed:.0f}s)", flush=True)
                    sys.exit(3)
                first = False
            with tb.step(f"{arm}:{i}"):
                row = run_item(arm, press, i)
            r0 = rows_by_arm.get("chunkkv_repro", {}).get(i) or repro_ref.get(i)
            if not arm.startswith("chunkkv_repro") and r0 is not None:
                assert row["kept_per_layer"] == r0["kept_per_layer"], f"G1 violated on item {i} arm {arm}"
                cnt["g1_vs_repro"] += 1
                assert abs(row["effective_cr"] - r0["effective_cr"]) < 1e-6, f"G2 violated on item {i} arm {arm}"
                cnt["g2_vs_repro"] += 1
            rows_by_arm[arm][i] = row
            cnt["n_rows"] += 1
            fout.write(json.dumps(row) + "\n"); fout.flush(); os.fsync(fout.fileno())
        if expired:
            break
        if a.g3 and arm == "chunkkv_repro":
            r0 = [rows_by_arm[arm][i] for i in items if i in rows_by_arm[arm]]
            assert len(r0) == len(items), "G3 needs the whole smoke id list"
            diff = float(np.mean([r["score"] for r in r0]) - np.mean([r["pub_score"] for r in r0]))
            match = float(np.mean([r["predicted_answer"].strip() == r["pub_pred"].strip() for r in r0]))
            per_task = {t: float(np.mean([r["score"] - r["pub_score"] for r in r0 if r["task"] == t])) for t in sorted({r["task"] for r in r0})}
            ok = abs(diff) <= a.g3_floor_pp and match >= a.g3_match
            gates.update(G3=dict(n=len(r0), pooled_diff_pp=diff, exact_match=match, floor_pp=a.g3_floor_pp, match_min=a.g3_match,
                                 per_task_diff_pp=per_task, repro_mean=float(np.mean([r["score"] for r in r0])),
                                 published_mean=float(np.mean([r["pub_score"] for r in r0]))), G3_pass=bool(ok))
            json.dump(gates, open(pre + "_gates.json", "w"), indent=1)
            print("G3", json.dumps(gates["G3"]), "pass" if ok else "FAIL", flush=True)
            if not ok:
                stop.set(); fout.close(); tb.finish()
                sys.exit(4)
    stop.set()
    fout.close()
    tbm = tb.finish()
    n_rows = sum(len(v) for v in rows_by_arm.values())
    gates.update(counts=cnt, n_rows_total=n_rows, partial=bool(expired),
                 timebox=dict(n_done=tbm["n_done"], wall_s=tbm["wall_s"]),
                 cuda_peak_gb=(torch.cuda.max_memory_allocated(dev) / 2**30 if a.device.startswith("cuda") else None))
    json.dump(gates, open(pre + "_gates.json", "w"), indent=1)
    for suffix in ("checkpoint.json",):
        try:
            os.remove(os.path.join(pre + "_tb", suffix))
        except OSError:
            pass
    os.replace(os.path.join(pre + "_tb", "timebox.json"), pre + "_timebox.json")
    print("DONE", json.dumps({"n_rows_total": n_rows, "partial": bool(expired), "counts": cnt}), flush=True)
    sys.exit(5 if expired else 0)


if __name__ == "__main__":
    main()
