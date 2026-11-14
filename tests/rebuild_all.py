# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Run every analysis script in order, then compare each value they write (out/<name>.json) with expected/<name>.json.

Usage: python tests/rebuild_all.py   (or ./rebuild_all.sh)
Exit status 0 if every expected value is reproduced within its tolerance. A value with a "known" note in expected/ that does not match is reported as known-deviation (listed, not fatal). A value flagged "mc" in expected/ (a bootstrap interval end) may
differ by up to twice the tolerance and is then counted as 'mc-noise'; the largest such deviation is reported. A full log is printed.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["periodogram", "period20", "pb3", "pb2", "pa2", "pa2_mechanism", "pa3", "c2g", "fig1_flagship", "table6_options"]
OUTNAME = {"table6_options": "table6"}
FIGS = ["fig1_flagship.png", "fig2_chunkkv_phase.png", "fig3_pb3.png", "fig4_iso.png", "fig5_c2g.png", "fig6_pb2.png"]


def main():
    t0 = time.time()
    os.makedirs(f"{ROOT}/out", exist_ok=True)
    for s in ORDER:
        t = time.time()
        r = subprocess.run([sys.executable, f"{ROOT}/analysis/{s}.py"], capture_output=True, text=True, cwd=ROOT)
        status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
        print(f"[run] analysis/{s}.py {status} in {time.time() - t:.0f} s")
        if r.returncode != 0:
            print(r.stderr[-1500:])
            sys.exit(2)
    total = {"match": 0, "mc-noise": 0, "known-deviation": 0, "mismatch": 0, "missing": 0}
    rows = []
    for name in sorted(f[:-5] for f in os.listdir(f"{ROOT}/expected") if f.endswith(".json")):
        exp = json.load(open(f"{ROOT}/expected/{name}.json"))
        got = json.load(open(f"{ROOT}/out/{name}.json"))
        c = {"match": 0, "mc-noise": 0, "known-deviation": 0, "mismatch": 0, "missing": 0}
        maxdev_mc = 0.0
        for k, e in sorted(exp.items()):
            if k not in got:
                c["missing"] += 1
                rows.append((name, k, e["value"], None, "missing"))
                continue
            g = got[k]
            if isinstance(e["value"], str):
                ok = str(g) == e["value"]
                st = "match" if ok else "mismatch"
            else:
                dev = abs(float(g) - e["value"])
                if dev <= e["tol"] + 1e-9:
                    st = "match"
                elif e.get("mc") and dev <= 2 * e["tol"] + 1e-9:
                    st = "mc-noise"
                    maxdev_mc = max(maxdev_mc, dev)
                else:
                    st = "known-deviation" if e.get("known") else "mismatch"
            c[st] += 1
            if st != "match":
                rows.append((name, k, e["value"], g, st))
        for k in c:
            total[k] += c[k]
        print(f"[check] {name}: {len(exp)} expected values: " + ", ".join(f"{k} {v}" for k, v in c.items()) + (f"; largest bootstrap-end deviation {maxdev_mc:.4g}" if maxdev_mc else ""))
    for r in rows:
        print("  not matched:", r)
    for f in FIGS:
        print(f"[figure] figures/{f}: {'present, %d bytes' % os.path.getsize(f'{ROOT}/figures/{f}') if os.path.exists(f'{ROOT}/figures/{f}') else 'MISSING'}")
    print("[total]", total, f"wall {time.time() - t0:.0f} s")
    json.dump({"total": total, "not_matched": [list(map(str, r)) for r in rows]}, open(f"{ROOT}/out/verification.json", "w"), indent=1)
    sys.exit(0 if total["mismatch"] == 0 and total["missing"] == 0 else 1)


if __name__ == "__main__":
    main()
