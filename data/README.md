# data/ (derived tables, CC BY 4.0)

Every file is a derived table. No generated answers, no benchmark texts, no raw API outputs.

| File | Content | Columns |
|---|---|---|
| ruler_needle_positions.csv | Token position of the queried key in the RULER-4096 contexts (six needle tasks, two tokenizers) | model, item, task, pos, ntok |
| leaderboard_item_correct.csv.gz | Per-item correctness (1 = all reference strings found) of the audited leaderboard entries on the six needle tasks | model, press, cr, item, y |
| leaderboard_item_scores_13task.csv.gz | String-match score (0-100) of the published ChunkKV entry, the track leader AdaKV-DropKV and the uncompressed model on all 13 tasks | model, entry, cr, item, task, score |
| leaderboard_sota_rows.csv | Reference rows read from the leaderboard metrics files (ChunkKV, best entry per track, uncompressed) with per-task scores | row_type, model, cr_nominal, track, press_name, ..., mean13 |
| periodogram_F_periods2to64.csv | F statistic of two harmonics per period (2-64) for 98 leaderboard entries (all presses with an estimable entry at CR 0.50, 0.75 or 0.88; one press of the leaderboard is not listed, see RELEASE_NOTES.md) | model, press, cr, acc, P, F, p |
| pb3/base_margins_{flash,pro}.csv | First-token margin (logit) of every probe at each of 16 shifts (j00-j15) | probe, source, j00..j15 |
| pb3/h3_per_probe_{flash,pro}.csv | Predicted and observed typo contrast per probe | probe, source, pred, obs |
| pb3/h4_per_probe_{flash,pro}.csv | Typo-induced margin change, single call and shift-averaged | probe, source, single, avg, single_vs_mean16 |
| pb3/keep_{flash,pro}.csv | Keep rule per probe (median margin of 16 repeat calls above 0) | model, probe, source, kept, repeat_median_margin, repeat_greedy_correct, n_repeat |
| pb3/noise_table_{flash,pro}.csv | Post hoc noise-control margins per probe | probe, source, single, avg1.., A_single, A_avg, M1.., B_single, B_avg |
| pb2/flip_per_probe_flash_1k.csv | Per-probe flip rates of the PB2 arms (Flash, 1k context) | probe, typo_d0, ..., repeat, odd, even, p1 |
| pb2/pb2_results.json | Registered PB2 summary statistics | nested |
| pa2/pa2_outcomes.csv.gz | Outcome per context and arm of run pa2 (0/1 correct; share of key/value tokens kept) | cr, arm, item, essay, phase_needle, correct, val_kept, needle_kept, needle_start |
| pa3/pa3_outcomes.csv.gz | Outcome, span geometry and selection timing per context and arm of run pa3 | src, ctx, cr, arm, correct, eff_cr, ..., t_sel, t_score |
| pa3/timing_*.jsonl | Timing of three contexts per mode on the test CPU | mode, cr, t_prefill_total, t_select_gather, ... |
| c2g/c2g_item_scores.csv.gz | Per-item score of the two-grid arm and of the local ChunkKV rerun with the published score, timing, and whether the rerun answer equals the published answer (full stage) | cell, arm, run_id, item_id, task, score, pub_score, same_answer, timing columns |
| c2g/c2g_smoke_item_scores.csv.gz | Same for the 195-item smoke stage (all arms) | cell, arm, item_id, task, score, pub_score, same_answer |

`cr` is the nominal compression ratio (0.88 stands for the leaderboard label of 0.875). `item` is the row index in the RULER-4096 benchmark (13 tasks x 500 items).
How each table was derived from the original files is documented in `data_build/` (reference scripts; they need the original files).
