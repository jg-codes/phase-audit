# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""pb3 harness (copy of the pb2 harness; changes: PB3_ROOT, row timestamp `ts`): phase-controlled single-token probes of DeepSeek chunked-KV models (defensive measurement; benign probes only).
Reuses the pb1 harness design (API read at temperature 1, max_tokens 1, top_logprobs 20, thinking disabled, beta assistant prefix).
Key is read from env DEEPSEEK_API_KEY and never printed. Tokenizer + official encoders from the HF repos
(deepseek-ai/DeepSeek-V4.1-Flash-1017, deepseek-ai/DeepSeek-V4-Pro) in $PB3_ROOT/tok_flash, tok_pro."""
import os, re, json, time, math, hashlib, datetime, threading, importlib.util
import requests
from concurrent.futures import ThreadPoolExecutor
from tokenizers import Tokenizer

ROOT = os.environ.get("PB3_ROOT", "pb3")
MODELS = {"flash": "deepseek-flash", "pro": "deepseek-v4-pro"}
STRIDE = {"flash": 2, "pro": 4}
M_CSA = {"flash": 2, "pro": 4}
PRICE = {"flash": (0.003, 0.15, 0.60), "pro": (0.022, 0.66, 1.98)}   # USD / 1M tokens off-peak: cache hit, miss, output
PEAK_HOURS = {1, 2, 3, 6, 7, 8, 9}                                  # UTC, Mon-Fri: x2
PAD = " pad"
_lock = threading.Lock()

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

TK = {n: Tokenizer.from_file(f"{ROOT}/tok_{n}/tokenizer.json") for n in ("flash", "pro")}
ENC = {"flash": _load(f"{ROOT}/tok_flash/encoding/encoding.py", "encf"), "pro": _load(f"{ROOT}/tok_pro/encoding/encoding_dsv4.py", "encp")}
SPECIAL_HEAD = {}

def sha(s): return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()

def render_user(model, user):
    return ENC[model].encode_messages([{"role": "user", "content": user}], thinking_mode="chat")

def tokenise(model, user, prefill):
    """full prompt string as the beta prefix API sees it (verified in pb1 against usage.prompt_tokens)"""
    s = render_user(model, user) + prefill
    return s, TK[model].encode(s, add_special_tokens=False)

def tok_index_of_char(enc, pos):
    for i, (a, b) in enumerate(enc.offsets):
        if a <= pos < b: return i, a
    raise ValueError("char not in any token")

BARRIER = {"flash": None, "pro": None}   # set by pb2_build (fixed 2m-token neutral barrier per model)

def build_user(P, model, pads=0, typo=None, barrier=False, suffix=0, ctx="1k"):
    """user message = host sentence (+typo) + pads + blank line + [barrier + blank line] + segment + suffix pads.
    returns user text, char offset of the segment start inside it, and typo char position inside the host."""
    host = P["host"]; tpos = None
    if typo is not None:
        w = P["typo"]["word"]; wi = P["typo"]["char_pos"]
        host = host[:wi] + P["typo"]["variants"][typo]["text"] + host[wi + len(w):]; tpos = wi
    head = host[:-1] + PAD * pads + host[-1] + "\n\n" + ((BARRIER[model] + "\n\n") if barrier else "")
    return head + P["segment"][ctx] + PAD * suffix, len(head), tpos

def prompt_info(P, model, ctx="1k", **kw):
    user, seg_off, tpos = build_user(P, model, ctx=ctx, **kw)
    s, e = tokenise(model, user, P["prefill"])
    base = s.index(user)                       # position of the user text inside the rendered string
    si, a = tok_index_of_char(e, base + seg_off)
    assert a == base + seg_off, "segment start is not a token boundary"
    info = dict(n_tokens=len(e.ids), start=si, typo_tok=(tok_index_of_char(e, base + tpos)[0] if tpos is not None else None))
    return user, info, s

def messages(user, prefill):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": prefill, "prefix": True}]

# ---------------- API ----------------
def _mult(ts=None):
    d = datetime.datetime.utcnow() if ts is None else ts
    return 2.0 if (d.weekday() < 5 and d.hour in PEAK_HOURS) else 1.0

def est_cost(model, u):
    hit, miss, out = PRICE[model]; pt = u.get("prompt_tokens", 0); ch = u.get("prompt_cache_hit_tokens", 0)
    cm = u.get("prompt_cache_miss_tokens", pt - ch); ct = u.get("completion_tokens", 0)
    return _mult() * (ch * hit + cm * miss + ct * out) / 1e6

def api(model, msgs, retries=5):
    key = os.environ["DEEPSEEK_API_KEY"]
    body = {"model": MODELS[model], "messages": msgs, "max_tokens": 1, "temperature": 1.0, "stream": False,
            "thinking": {"type": "disabled"}, "logprobs": True, "top_logprobs": 20}
    err = None
    for a in range(retries):
        try:
            r = requests.post("https://api.deepseek.com/beta/chat/completions", headers={"Authorization": "Bearer " + key}, json=body, timeout=180)
            if r.status_code == 200: return r.json()
            err = f"{r.status_code} {r.text[:160]}"
            if r.status_code in (401, 402, 403): break      # auth / balance: do not hammer
        except Exception as ex:
            err = repr(ex)[:160]
        time.sleep(min(30, 2 ** a))
    return {"error": err}

def balance():
    r = requests.get("https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"]}, timeout=30)
    return float(r.json()["balance_infos"][0]["total_balance"])

def read_rows(path):
    out = []
    if os.path.exists(path):
        for l in open(path):
            try: out.append(json.loads(l))
            except Exception: pass
    return out

def run_jobs(jobs, out_path, workers=12, cap_usd=None, spent=None, progress_every=500):
    """jobs: dict(id, model, user, prefill, meta). Appends rows; skips ids already present (resumable). Stops when ledger >= cap_usd."""
    done = {r["id"] for r in read_rows(out_path) if "dist" in r}
    todo = [j for j in jobs if j["id"] not in done]
    spent = spent if spent is not None else {"usd": 0.0, "n": 0, "err": 0}
    stop = threading.Event()
    def work(j):
        if stop.is_set(): return
        r = api(j["model"], messages(j["user"], j["prefill"]))
        row = {"id": j["id"], "model": j["model"], "meta": j["meta"], "ts": datetime.datetime.utcnow().isoformat() + "Z"}
        if "error" in r: row["error"] = r["error"]
        else:
            c = r["choices"][0]; lp = c["logprobs"]["content"][0]["top_logprobs"] if c.get("logprobs") and c["logprobs"].get("content") else []
            row.update(dist=[(x["token"], math.exp(x["logprob"])) for x in lp], lps=[(x["token"], x["logprob"]) for x in lp],
                       text=c["message"].get("content"), usage=r["usage"], cost=est_cost(j["model"], r["usage"]))
        with _lock:
            open(out_path, "a").write(json.dumps(row) + "\n")
            spent["usd"] += row.get("cost", 0); spent["n"] += 1; spent["err"] += int("error" in row)
            if cap_usd is not None and spent["usd"] >= cap_usd: stop.set()
            if spent["err"] >= 40 and spent["err"] > 0.2 * spent["n"]: stop.set()
            if spent["n"] % progress_every == 0: print(f"[{datetime.datetime.utcnow():%H:%M:%S}] n={spent['n']} usd={spent['usd']:.3f} err={spent['err']}", flush=True)
    with ThreadPoolExecutor(workers) as ex: list(ex.map(work, todo))
    return spent, len(todo), stop.is_set()

def greedy_tok(dist): return dist[0][0] if dist else None
def is_gold(tok, gold): return tok is not None and tok.strip() == gold.strip()
