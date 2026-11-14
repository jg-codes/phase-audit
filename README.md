# phase-audit: code and data of "Offset blind spots in compressed language models: a black-box audit and a two-grid remedy"

Version 1.0.0, archived at https://doi.org/10.5281/zenodo.24229081 (all versions). Author: Johannes Gast, Ingenieurbuero Johannes Gast (independent researcher).

## What the paper claims

1. A periodogram over the per-item predictions that the kvpress leaderboard already publishes recovers the configured block size of ChunkKV (20 tokens), CUR (16) and LagKV (128) in the strongest entry of each press, and misses the 256-token blocks of Compactor.
2. ChunkKV's period-20 effect persists after controlling for needle position; the unchunked SnapKV twin with the same window shows no period-20 effect in four of its five estimable entries.
3. A pre-registered test on 120 fresh probes recovers the compression stride of DeepSeek-V4.1-Flash (2) and DeepSeek-V4-Pro (4) from first-token margins (3.11 and 2.74 times the permutation null); a registered flip-rate law that preceded it was refuted.
4. Two-grid selection (ChunkKV chunks chosen across two grids offset by half a chunk, same token budget) raises the RULER-4096 13-task mean over the published ChunkKV entry by 5.74 (Qwen3-8B) and 2.01 (Llama-3.1-8B-Instruct) points at CR 0.75 and by 10.27 points for Qwen3-8B at CR 0.875, with losses on variable tracking and common-word extraction at CR 0.875.
5. The Llama-3.1-8B-Instruct cell at CR 0.875 was stopped by its smoke-stage kill rule; the open-model runs (Qwen3-0.6B) are exploratory.

## Install and rebuild

```
python -m pip install -r requirements.txt     # Python 3.11; versions as tested
./rebuild_all.sh                              # runs every analysis script and compares with expected/ (about 90 s on 8 CPUs)
```

`rebuild_all.sh` writes values to `out/`, figures to `figures/`, and exits with status 0 if every value in `expected/` is reproduced within its tolerance (half a unit of the last printed digit; p-values within 6%). `expected/` holds 695 values parsed from the printed tables and from numbers stated in the text (`tools/make_expected.py`). Seeds are fixed. Verified result of the last full run in a clean environment: see `REPRO_LOG.txt` in the release assets (693 values match, 2 known deviations, 0 mismatches).

One command per table or figure (each reads `data/` only):

| Paper item | Command |
|---|---|
| Table 1, Table 7 (Appendix B), wider-leaderboard median, Figure 1a caption value | `python analysis/periodogram.py` |
| Table 2, Figure 2 | `python analysis/period20.py` |
| Table 3, Figure 3, Figure 1b, noise controls | `python analysis/pb3.py` |
| PB2 numbers of Results 4 and Appendix A, Figure 6 | `python analysis/pb2.py` |
| Table 4, pa2 numbers of Results 5 | `python analysis/pa2.py` |
| Span geometry (AUC 0.916, 96.2%, 4 of 20 offsets) | `python analysis/pa2_mechanism.py` |
| Figure 4, pa3 numbers, selection timing on the test CPU | `python analysis/pa3.py` |
| Table 5, Appendix C (per-task scores, selection cost), Appendix A numbers of C2G, Figure 5 | `python analysis/c2g.py` |
| Figure 1 (three panels; run `pb3.py` and `pa2.py` first) | `python analysis/fig1_flagship.py` |
| Table 6 (values gathered from the scripts above) | `python analysis/table6_options.py` |

Unit test of the press: `python -m pytest tests/test_two_grid_chunkkv_press.py` (needs `requirements-press.txt`; 210 cases pass with kvpress 0.5.5).

## What is included

- `src/`: the two-grid ChunkKV press (`two_grid_chunkkv_press.py`) and reference copies of the runners (see `src/README.md`).
- `analysis/`: one script per table and figure, plus the statistics modules (`pb3_stats.py` and `c2g_stats.py` are the registered statistics of PB3 and C2G; `pa2_stats.py` and `pa3_stats.py` belong to the exploratory open-model runs).
- `data/`: derived tables (per-item correctness and scores, per-probe margins, summaries); dictionary in `data/README.md`.
- `expected/`: the values each script must reproduce, as json. `figures/`: rebuilt PNGs. `tests/`: `rebuild_all.py` and the unit test.
- `data_build/`: reference scripts that derive `data/` from the original files and the public leaderboard (they need files that are not part of this release).
- `.github/workflows/rebuild.yml`: runs `./rebuild_all.sh` on every push (actions pinned by commit SHA).

## What is not included

- Registration files. They are available from the author on request. SHA-256 as printed in Appendix A of the paper:
  - PB2: `5c99cddeca886677cf5c26f3ed2fbf8596deffab46feb0d4360d4cec3cab8438`
  - PB3: `720d310c3995b82dc15939ce22fbbfb15d31ed7cf0a7d836354c6eb7f389bf71`
  - C2G: `dba21471ffca7a0bb8a165c5a4138ea0d21eb5689fb1a90c1499966e793bc19e`
  - C2G smoke-stage addendum: `feff8951010727092e74e859725518714618359d3cf0940066ec1c8671bcfe22`
- Raw DeepSeek API outputs (available from the author on request), benchmark texts (RULER), model weights, and the leaderboard predictions (public: Hugging Face Space `nvidia/kvpress-leaderboard`, revision `e81fc403`, download script `data_build/download_leaderboard.py`).
- Row files of the open-model runs with generated answers (about 55 MB for C2G): listed with names, sizes and SHA-256 in `DATA_RECORD.md`.
- The GPU and API runs themselves. The runners are reference copies. The original runs used a kvpress source tree packed with the run, not a PyPI release. The press (unit test) and the kvpress modules imported by the pa1, pa2 and pa3 runners were checked against public kvpress 0.5.5. `src/c2g_runner.py` imports the press as `kvpress.presses.two_grid_chunkkv_press`, so `src/two_grid_chunkkv_press.py` must first be copied into `kvpress/presses/` (as `src/c2g_job.sh` does); with that copy the import resolves against 0.5.5 (tested by import only). No run was repeated on it.

## Not rebuilt by this release

1. PB1 exploratory count (33 of 48 sentences): raw rows not released, only a stored summary existed.
2. PB2: only the parity contrast P1 and the repeat-call flip rate of Flash are recomputed (from the per-probe table). Figure 6 is drawn from the registered summary `data/pb2/pb2_results.json`; the verdicts for P2 to P4 and the panels (c) and (d) are not recomputed from raw rows.
3. PB3 from raw API calls: H1 to H4 and the noise controls are rebuilt from the derived per-probe tables (margins), not from the raw calls.
4. Choice of the track leader and of the uncompressed reference: read from `data/leaderboard_sota_rows.csv` (derived from the leaderboard metrics files), not recomputed over the whole leaderboard.
5. Median ratio over the wider leaderboard (Results 1) uses the derived F table of 82 other entries (`data/periodogram_F_periods2to64.csv`); the per-item data of those entries are not released.
6. Registration contents and outcomes (Appendix A text), the cost figures (USD 55 and 61 to 62), the numbers of Periodic Weak Spots (Zhu et al.), and the smoke-stage exploratory arms (swin, three-grid, fine-10, lazy) are not rebuilt.
7. Figures are rebuilt from their data and checked by eye against the paper's figure files for Figures 2 and 4; they differ in fonts and sizes. Figure 1 panel c and Figure 5 panel a carry the corrected titles proposed in the independent review.
8. Table 6 reuses values of other tables; its latency entry for averaged API calls ("not measured") is a statement, not a number.

## Known deviations

- Two timing medians of Appendix C (Qwen3-8B, CR 0.875, two-grid arm: decode 0.609 and total 1.121 s in the paper) come out as 0.610 and 1.120 s from the released per-item table. The difference is 0.001 s and depends on which rows count as the first three of a run.
- The weak-offset correlation in Results 5 is printed once (0.98); the rebuild gives 0.984 at CR 0.75 and 0.991 at CR 0.875.
- The audit set uses CUR at CR 0.75 and 0.88 only, as in the paper (CR 0.50 entries of CUR are not part of the 16 blocked entries).

## AI assistance

Code and analyses were written and run with Claude Science of Anthropic, an agentic research environment with code execution, under the author's direction, in October and November 2026. This release bundle was assembled by the same assistant. The author designed and directed the research and is responsible for all claims.

## Licence and citation

Copyright 2026 Johannes Gast. Code: Apache-2.0 (`LICENSE`). Data tables, expected values and figures: CC BY 4.0 (`LICENSE-DATA`).

Both licences allow use, change and redistribution, including commercial use, as long as the credit is kept: redistributions carry the `NOTICE` file and the copyright lines, and reuse of data or figures names the author and this release. Please cite the paper and the release as given in `CITATION.cff`. The two-grid press subclasses `ChunkKVPress` of kvpress (NVIDIA, Apache-2.0), so the same licence applies upstream and downstream. For other terms, contact the author.
