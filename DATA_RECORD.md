# Files for a separate data record

These files are too large for the repository (about 55 MB for the C2G rows alone; the limit for this repository is 20 MB per file and 50 MB in total) and contain generated answer strings of the open models. The repository holds compact derived tables instead (`data/`). The files below would go into a separate data record, cross-linked with the repository, if the author decides to publish them. Sizes and SHA-256 are those of the files in the original project store.

## Open-model row files (candidates for the data record)

| File (original name) | Size in bytes | SHA-256 |
|---|---|---|
| c2gF_llama_75_chunkkv_repro_ALL_rows.jsonl | 9,684,613 | `1d8b4cd7874ca81d278d894f98c9c16cd6ef9ce5696fb21f005a79eea1cf3188` |
| c2gF_llama_75_two_grid_ALL_rows.jsonl | 9,203,504 | `5625642efa6ae77bf8421d976413225df71723ea66c9db215aba196a1e7a784f` |
| c2gF_qwen_75_chunkkv_repro_ALL_rows.jsonl | 10,422,067 | `8a7888132d78ea041390ba8f583362ae6efc9850bbd22348f66c144e2dd5baa4` |
| c2gF_qwen_75_two_grid_ALL_rows.jsonl | 9,883,843 | `f358dd82835d79a1d7619dae10dba12f8b6ab1fa8d39957197c1eab6d69893b9` |
| c2gF_qwen_875_chunkkv_repro_ALL_rows.jsonl | 4,608,239 | `c21fe826b086302f9a2df5f90a8fe9386d558902d0603537cd2bf50d87a25d6d` |
| c2gF_qwen_875_two_grid_ALL_rows.jsonl | 9,919,207 | `001f3f172958c71d74685ce95d52604c05a50e7ee50de1337e1b206b1b975f26` |
| c2gS_llama_75_rows.jsonl | 852,892 | `4cadb73663d340033ed7aa36447cca40db0ceaaa3208094d3d97998c18cd5113` |
| c2gS_llama_875_rows.jsonl | 856,720 | `ece318ede98370a01a5a79c0bacc4fe0a4e4500ae74f5d5d0d325660b1d54a2f` |
| c2gS_qwen_75_rows.jsonl | 909,785 | `29d45a8cb68cab11127d23e63557e05f5b6c85765a4a67ee85d4276462181fcd` |
| c2gS_qwen_875_rows.jsonl | 2,112,384 | `94feb5e0716f1732601ff8e8c90f7f76a264fd05e4b42ac534cbb10753c3fc4f` |
| pa2_rows_synth.jsonl | 6,810,602 | `5ed62b7eb705d0d85d5dbc007618dd17e047a6f078ef1a4a8aa1a31a5efc7a90` |
| pa3_mainA_rows.jsonl | 2,164,913 | `2a051db78d8d91e78953bde32b1399a22213c90e357abfcd2c7877c02c393360` |
| pa3_mainB_rows.jsonl | 2,159,310 | `66b2546e89199ff22cddc51609fde466a867e494a02a61ee5c83a3bcdf36d20f` |

## Not released

The raw rows of the DeepSeek API tests (request and response rows, completions, log-probability lists) are not released and are available from the author on request. Files, for reference:

| File (original name) | Size in bytes | SHA-256 |
|---|---|---|
| pb2_rows_flash_cal.jsonl | 13,392,023 | `038ef41367a038625de0ab64f4be562f442cf58bbaf9b7569ef138d5c4dc79c4` |
| pb2_rows_flash_held.jsonl | 17,084,374 | `0fbdf400da10bcefd2651c8925dbae0714113be944f8fb41194a29a74c59b84d` |
| pb2_rows_flash_8k.jsonl | 768,540 | `466cb84335d4a7637362de9f3e03a8ae50aea28942123c31eabd603d17baee53` |
| pb2_rows_pro_cal.jsonl | 20,742,898 | `aeee75ee7ea5d19c59ebd5e8fe2eba19871c4780f285c1dd71f861e14333204d` |
| pb2_rows_pro_held.jsonl | 20,504,837 | `c3eae7782b3d15360a00602c4adc216e9cc1e838a68c5e7ac2c5c7f42c6234ef` |
| pb3_rows_flash.jsonl | 10,522,011 | `d4fcdf6255d87e47f3e4fe7daeaa0cbc561766b99e00b9b147a9bcd1dc8e3658` |
| pb3_rows_pro.jsonl | 12,798,445 | `d0a82fa1cdefb180b0d4e35f88618c07ebb8fefcebba266534f31e332a1695fd` |

The derived per-probe margin tables of PB2 and PB3 in `data/pb2/` and `data/pb3/` are released.

## Public inputs (not copied)

- kvpress leaderboard predictions and metrics: Hugging Face Space `nvidia/kvpress-leaderboard`, revision `e81fc403a50302632134b2a350f0da8ab5bfa6f5`. Download script: `data_build/download_leaderboard.py`.
- RULER-4096 benchmark (kvpress benchmark dataset, 6,500 items) and the tokenizers of Qwen/Qwen3-8B and meta-llama/Llama-3.1-8B-Instruct (gated): needed only by `data_build/build_leaderboard_tables.py`.
