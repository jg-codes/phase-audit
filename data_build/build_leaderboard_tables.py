# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""Build the derived per-item tables of the leaderboard-based analyses (Tables 1, 2, 5, 7; Figures 1a, 2, 5).

Inputs (not part of this repository, see DATA_RECORD.md):
  --ruler   RULER-4096 parquet of the kvpress benchmark (6,500 items, 13 tasks x 500, columns task, context, question, answer)
  --tok-qwen, --tok-llama   tokenizer.json files of Qwen/Qwen3-8B and meta-llama/Llama-3.1-8B-Instruct
  --lb      directory written by download_leaderboard.py (Space revision e81fc403)
Outputs in --out (default data/):
  ruler_needle_positions.csv        model, item, task, pos, ntok  (token position of the queried key, six needle tasks)
  leaderboard_item_correct.csv.gz   model, press, cr, item, y      (1 = answer string-matched in full, six needle tasks)
  leaderboard_item_scores_13task.csv.gz  model, entry, cr, item, task, score (0-100, string match score, 13 tasks)
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

TASKS6 = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multivalue"]
MODELS = {"Qwen--Qwen3-8B": "Qwen3-8B", "meta-llama--Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct"}
NP_PATTERN = re.compile(r"[\x00-\x1f]")


def item_score(task, pred, answers):
    pred = NP_PATTERN.sub("", str(pred).strip()).strip().lower()
    refs = [r.lower() for r in answers]
    if task.split("_")[0] == "qa":
        return 100.0 * max(1.0 if r in pred else 0.0 for r in refs)
    return 100.0 * sum(1.0 if r in pred else 0.0 for r in refs) / len(refs)


def leaderboard_dir(lb, m, press, cr, qa):
    return os.path.join(lb, "benchmark", f"ruler__4096__{m}__{press}__{cr}" + ("__query_aware" if qa else ""), "predictions.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler"); ap.add_argument("--tok-qwen"); ap.add_argument("--tok-llama")
    ap.add_argument("--lb"); ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    toks = {"Qwen--Qwen3-8B": Tokenizer.from_file(a.tok_qwen), "meta-llama--Llama-3.1-8B-Instruct": Tokenizer.from_file(a.tok_llama)}
    RU = pd.read_parquet(a.ruler)
    RU["item"] = range(len(RU))
    answers = [list(x) for x in RU["answer"]]
    RU["key"] = RU.question.str.extract(r"special magic (?:number|numbers|uuid) for (.+?) mentioned")[0]

    # 1. needle positions (six needle tasks, both tokenizers)
    recs = []
    for _, row in RU[RU.task.isin(TASKS6)].iterrows():
        m_ = re.search(r"for " + re.escape(row.key) + r" is", row.context)
        pre = row.context[: m_.start() + 4]
        for mn, tk in toks.items():
            recs.append(dict(model=MODELS[mn], item=row["item"], task=row.task,
                             pos=len(tk.encode(pre, add_special_tokens=False).ids),
                             ntok=len(tk.encode(" " + row.key, add_special_tokens=False).ids)))
    pd.DataFrame(recs).to_csv(os.path.join(a.out, "ruler_needle_positions.csv"), index=False)

    # 2. per-item correctness of the audited entries (six needle tasks)
    audited = [("chunkkv__Q", "chunkkv", True), ("snapkv__Q", "snapkv", True), ("cur", "cur", False), ("lagkv", "lagkv", False),
               ("adakv_compactor", "adakv_compactor", False), ("knorm", "knorm", False), ("streaming_llm", "streaming_llm", False)]
    rows = []
    for m in MODELS:
        for name, tag, qa in audited + [("no_press", "no_press", False)]:
            for cr in (["0.00"] if name == "no_press" else ["0.75", "0.88"] if name == "cur" else ["0.50", "0.75", "0.88"]):
                p = leaderboard_dir(a.lb, m, tag, cr, qa)
                if not os.path.exists(p):
                    continue
                d = pd.read_csv(p, dtype=str, keep_default_na=False)
                assert len(d) == 6500 and (d.task.values == RU.task.values).all()
                y = [item_score(RU.task.iat[i], d.predicted_answer.iat[i], answers[i]) >= 100 - 1e-9 for i in range(6500)]
                y = np.array(y)
                keep = RU.task.isin(TASKS6).values
                rows.append(pd.DataFrame(dict(model=MODELS[m], press=name, cr=float(cr), item=np.nonzero(keep)[0], y=y[keep].astype(int))))
    pd.concat(rows).to_csv(os.path.join(a.out, "leaderboard_item_correct.csv.gz"), index=False)

    # 3. 13-task item scores of the entries compared in the registered benchmark test
    rows = []
    for m in MODELS:
        for entry, tag, qa, crs in [("chunkkv", "chunkkv", True, ["0.75", "0.88"]), ("leader_adakv_dropkv", "adakv_dropkv", True, ["0.75", "0.88"]),
                                    ("no_press", "no_press", False, ["0.00"])]:
            for cr in crs:
                p = leaderboard_dir(a.lb, m, tag, cr, qa)
                d = pd.read_csv(p, dtype=str, keep_default_na=False)
                sc = [item_score(RU.task.iat[i], d.predicted_answer.iat[i], answers[i]) for i in range(6500)]
                rows.append(pd.DataFrame(dict(model=MODELS[m], entry=entry, cr=float(cr), item=range(6500), task=RU.task.values, score=sc)))
    pd.concat(rows).round(4).to_csv(os.path.join(a.out, "leaderboard_item_scores_13task.csv.gz"), index=False)
    print("done")


if __name__ == "__main__":
    main()
