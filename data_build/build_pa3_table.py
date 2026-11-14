# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Derive the compact outcome table of run pa3 (Qwen3-0.6B, CPU, iso-memory sweep) from the two runner row files (not released).
Usage: python data_build/build_pa3_table.py A=pa3_mainA_rows.jsonl B=pa3_mainB_rows.jsonl data/pa3/pa3_outcomes.csv.gz
Keeps numeric outcome, span-geometry and timing columns; drops generated text and masks."""
import json
import sys

import pandas as pd

frames = []
for spec in sys.argv[1:-1]:
    label, path = spec.split("=", 1)
    d = pd.DataFrame([json.loads(line) for line in open(path) if line.strip()])
    d["src"] = label
    frames.append(d)
D = pd.concat(frames, ignore_index=True)
cols = ["src", "ctx", "mode", "cr", "arm", "correct", "eff_cr", "needle_start", "item", "essay", "phase_needle", "kv_full", "val_full",
        "val_kept", "needle_kept", "key_kept", "kv_kept", "t_sel", "t_score", "t_prefill_rec", "t_dec", "n_gen", "verified_ctx", "load1"]
D[[c for c in cols if c in D.columns]].to_csv(sys.argv[-1], index=False)
print(len(D), "rows")
