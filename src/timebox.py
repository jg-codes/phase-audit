# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0
"""timebox: per-item timing, ETA and hard deadline with partial-result return.

Usage in a run script:
    from timebox import TimeBox
    tb = TimeBox(total=len(items), time_limit_s=args.time_limit, out_dir="out")
    for item in items:
        if tb.expired(): break
        with tb.step(item.id):
            result = process(item)
            tb.checkpoint(item.id, result)      # json-serialisable
    tb.finish()   # writes out/timebox.json: n_done, n_total, partial, per-item seconds, ETA history
"""
import json, os, time, contextlib

class TimeBox:
    def __init__(self, total, time_limit_s=None, out_dir="out", eta_after=3, log=print):
        self.total, self.limit, self.out, self.eta_after, self.log = total, time_limit_s, out_dir, eta_after, log
        self.t0 = time.time(); self.times = {}; self.results = {}
        os.makedirs(out_dir, exist_ok=True)
    def elapsed(self): return time.time() - self.t0
    def expired(self):
        if self.limit is None: return False
        # stop early if the *next* item would not fit
        mean = (sum(self.times.values()) / len(self.times)) if self.times else 0.0
        return self.elapsed() + mean > self.limit
    def eta_s(self):
        if len(self.times) < self.eta_after: return None
        mean = sum(self.times.values()) / len(self.times)
        return mean * (self.total - len(self.times))
    @contextlib.contextmanager
    def step(self, key):
        t = time.time()
        try: yield
        finally:
            self.times[key] = time.time() - t
            eta = self.eta_s()
            self.log(f"[timebox] {len(self.times)}/{self.total} {key} {self.times[key]:.1f}s"
                     + (f" ETA {eta/60:.1f} min" if eta is not None else "")
                     + (f" limit {self.limit/60:.0f} min" if self.limit else ""))
    def checkpoint(self, key, result):
        self.results[key] = result
        with open(os.path.join(self.out, "checkpoint.json"), "w") as f: json.dump(self.results, f)
    def finish(self):
        n = len(self.times)
        meta = {"n_done": n, "n_total": self.total, "partial": n < self.total, "wall_s": self.elapsed(),
                "per_item_s": self.times, "mean_item_s": (sum(self.times.values())/n if n else None), "time_limit_s": self.limit}
        with open(os.path.join(self.out, "timebox.json"), "w") as f: json.dump(meta, f, indent=1)
        self.log(f"[timebox] done {n}/{self.total} in {self.elapsed()/60:.1f} min" + (" (PARTIAL)" if meta["partial"] else ""))
        return meta
