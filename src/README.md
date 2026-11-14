# src/ (reference copies)

`two_grid_chunkkv_press.py` is the two-grid ChunkKV press of the paper. It subclasses `ChunkKVPress` of the public kvpress package; its unit test
(`tests/test_two_grid_chunkkv_press.py`, 210 cases) passes with kvpress 0.5.5 and torch 2.15.0 on CPU (`pip install -r requirements-press.txt`, then `python -m pytest tests/test_two_grid_chunkkv_press.py`).

The runners are reference copies. They need GPU or model files and were not re-run for this release. The kvpress modules they import resolve against kvpress 0.5.5 (tested by import only). `c2g_runner.py` imports the press as `kvpress.presses.two_grid_chunkkv_press`: copy `two_grid_chunkkv_press.py` into `kvpress/presses/` of the installed package first (as `c2g_job.sh` does); without that copy the import fails.
- `c2g_runner.py`, `timebox.py`, `c2g_job.sh`: registered benchmark test C2G on RULER-4096 (full stage, A100). `c2g_job.sh` is the Modal job wrapper of the original run; it unpacks a kvpress payload tarball that is not part of this release.
- `pa2_runner.py`, `pa3_runner.py`: open-model runs pa2 and pa3 (Qwen3-0.6B, CPU). `pa1_runner.py`: first exploratory run (named in the Method section, no result quoted).
- `pb3_harness.py`, `pb3_run.py`: client of the registered DeepSeek API test PB3. They read the API key from the environment variable DEEPSEEK_API_KEY and need the frozen probe files, which are not included.
