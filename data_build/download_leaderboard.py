# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Download the public kvpress leaderboard files used in the paper, pinned to the Space revision e81fc403.

Usage: python data_build/download_leaderboard.py LB_DIR
Needs `huggingface_hub`; a Hugging Face token (env HF_TOKEN) is optional.
Downloads predictions.csv for the entries used by the audit and by the registered benchmark test,
and metrics.json for every RULER-4096 entry (for the leaderboard summary table).
"""
import os
import sys
from huggingface_hub import HfApi, hf_hub_download

REPO = "nvidia/kvpress-leaderboard"
REV = "e81fc403a50302632134b2a350f0da8ab5bfa6f5"
MODELS = ["Qwen--Qwen3-8B", "meta-llama--Llama-3.1-8B-Instruct"]
# (press directory tag, query-aware suffix, compression ratios)
PRED_ENTRIES = [
    ("chunkkv", True, ["0.50", "0.75", "0.88"]),
    ("snapkv", True, ["0.50", "0.75", "0.88"]),
    ("cur", False, ["0.50", "0.75", "0.88"]),
    ("lagkv", False, ["0.50", "0.75", "0.88"]),
    ("adakv_compactor", False, ["0.50", "0.75", "0.88"]),
    ("knorm", False, ["0.50", "0.75", "0.88"]),
    ("streaming_llm", False, ["0.50", "0.75", "0.88"]),
    ("adakv_dropkv", True, ["0.75", "0.88"]),
    ("no_press", False, ["0.00"]),
]


def main(out):
    tok = os.environ.get("HF_TOKEN")
    api = HfApi(token=tok)
    files = set(api.list_repo_files(REPO, repo_type="space", revision=REV))
    want = []
    for m in MODELS:
        for press, qa, crs in PRED_ENTRIES:
            for cr in crs:
                d = f"benchmark/ruler__4096__{m}__{press}__{cr}" + ("__query_aware" if qa else "")
                if d + "/predictions.csv" in files:
                    want.append(d + "/predictions.csv")
                else:
                    print("missing", d)
    want += [f for f in files if f.startswith("benchmark/ruler__4096__") and f.endswith("/metrics.json")]
    for f in sorted(want):
        hf_hub_download(REPO, f, repo_type="space", revision=REV, token=tok, local_dir=out)
    print("downloaded", len(want), "files to", out)


if __name__ == "__main__":
    main(sys.argv[1])
