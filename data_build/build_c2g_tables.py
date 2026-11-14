# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Derive the per-item score tables of the registered benchmark test C2G from the runner row files (not released: about 55 MB, contain
generated answers). Usage: python data_build/build_c2g_tables.py ROWS_DIR OUT_DIR
ROWS_DIR holds c2gF_<cell>_<arm>_ALL_rows.jsonl (full stage) and c2gS_<cell>_rows.jsonl (smoke stage).
Outputs: c2g_item_scores.csv.gz (full stage), c2g_smoke_item_scores.csv.gz (smoke stage). Generated answers are reduced to one flag:
'same_answer' = the rerun answer string equals the published answer string."""
import glob
import json
import os
import sys

import pandas as pd

rows_dir, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
KEEP = ["arm", "item_id", "task", "score", "pub_score", "effective_cr", "score_s", "select_s", "prefill_s", "decode_s", "total_s", "peak_mem_gb"]


def read(paths, stage):
    recs, seen = [], set()
    for p in paths:
        cell = os.path.basename(p).split("_")[1] + "_" + os.path.basename(p).split("_")[2]
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["run_id"], r["arm"], r["item_id"])
            if k in seen:
                continue
            seen.add(k)
            d = {c: r.get(c) for c in KEEP}
            d["cell"] = cell
            d["run_id"] = r["run_id"]
            d["model"] = r["model"]
            d["cr_nominal"] = r["cr_nominal"]
            d["same_answer"] = int(str(r.get("predicted_answer")) == str(r.get("pub_pred")))
            recs.append(d)
    return pd.DataFrame(recs)


full = read(sorted(glob.glob(f"{rows_dir}/c2gF_*_ALL_rows.jsonl")), "F")
full.to_csv(f"{out}/c2g_item_scores.csv.gz", index=False)
smoke = read(sorted(glob.glob(f"{rows_dir}/c2gS_*_rows.jsonl")), "S")
smoke.drop(columns=["score_s", "select_s", "prefill_s", "decode_s", "total_s", "peak_mem_gb"]).to_csv(f"{out}/c2g_smoke_item_scores.csv.gz", index=False)
print(full.groupby(["cell", "arm"]).size().to_string())
print(smoke.groupby(["cell", "arm"]).size().to_string())
