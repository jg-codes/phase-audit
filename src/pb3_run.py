# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""pb3 stage runner. Usage: python pb3_run.py flash|pro   (needs DEEPSEEK_API_KEY in env; never printed)
Refuses to start if the frozen files / prereg hash do not match pb3_frozen.json. Balance is read before and after each stage (pb3_balance_log.csv);
the ledger (usage-based) stops the stage at LEDGER_STOP; hard cap USD 15 on total spend."""
import os, sys, json, glob, hashlib, datetime
os.environ.setdefault("PB3_ROOT", "pb3")
import pb3_harness as H, pb3_jobs as J
HARD_CAP_USD = 15.0; LEDGER_STOP = 13.5
PREREG_SHA = "720d310c3995b82dc15939ce22fbbfb15d31ed7cf0a7d836354c6eb7f389bf71"
def sha_file(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def check_frozen():
    fr = json.load(open("pb3_frozen.json"))
    for f, h in fr["files"].items():
        assert sha_file(f) == h, f"frozen file changed: {f}"
    assert fr["prereg_sha256"] == PREREG_SHA and sha_file("prereg_PB3_confirm.json") == PREREG_SHA
    return fr

def ledger_total(): return sum(r.get("cost", 0.0) for f in glob.glob("pb3_rows_*.jsonl") for r in H.read_rows(f))

def log_balance(tag):
    b = H.balance(); new = not os.path.exists("pb3_balance_log.csv")
    with open("pb3_balance_log.csv", "a") as f:
        if new: f.write("utc,tag,balance_usd\n")
        f.write(f"{datetime.datetime.utcnow().isoformat()}Z,{tag},{b}\n")
    return b

def run_stage(model):
    check_frozen(); probes = [json.loads(l) for l in open("pb3_probes.jsonl")]; jobs = []
    for P in probes: jobs += J.jobs_for_probe(P, model)
    jobs.sort(key=lambda j: (j["prio"], j["meta"]["probe"], j["id"])); out = f"pb3_rows_{model}.jsonl"
    bal0 = log_balance(f"{model}_start"); spent = {"usd": ledger_total(), "n": 0, "err": 0}
    print(f"stage {model}: {len(jobs)} jobs, ledger so far {spent['usd']:.3f} USD, balance {bal0}", flush=True)
    for pr in sorted({j["prio"] for j in jobs}):
        batch = [j for j in jobs if j["prio"] == pr]
        s, n, stopped = H.run_jobs(batch, out, workers=16, cap_usd=LEDGER_STOP, spent=spent)
        print(f"prio {pr}: {len(batch)} jobs; ledger {spent['usd']:.3f}; errors {spent['err']}; stopped={stopped}", flush=True)
        if stopped: break
    bal1 = log_balance(f"{model}_end"); print(f"stage {model} done: ledger {spent['usd']:.3f} USD; balance {bal0} -> {bal1}", flush=True)

if __name__ == "__main__":
    run_stage(sys.argv[1])
