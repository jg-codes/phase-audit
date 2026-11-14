# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Derive the compact per-context outcome table of run pa2 (Qwen3-0.6B, CPU) from the runner's row file (not released: large).
Usage: python data_build/build_pa2_table.py pa2_rows_synth.jsonl data/pa2/pa2_outcomes.csv.gz
Keeps only outcome columns (no generated text, no masks)."""
import json
import sys

import pandas as pd

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
d = pd.DataFrame(rows)
d = d[d["mode"] == "synth"]
cols = ["cr", "arm", "item", "essay", "phase_needle", "correct", "val_kept", "needle_kept", "needle_start"]
d[cols].to_csv(sys.argv[2], index=False)
print(len(d), "rows")
