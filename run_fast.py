"""Fast restartable runner: one persistent worker pool, every finished replication appended to disk at once,
largest projects first. Random-number streams are identical to runner.run (same seed per project and replication).

usage: python run_fast.py SPLIT_FILE SET SEED_SPLIT REPS "OFFSETS" SPECS_JSON OUT_DIR
"""
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import runner  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def done_reps(path):
    if not path.exists():
        return set()
    seen = {}
    with open(path, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            seen.setdefault(int(r["rep"]), set()).add(r["method"])
    return {rep for rep, ms in seen.items() if len(ms) >= NMETH}


if __name__ == "__main__":
    split_file, set_name, seed_split, reps, offsets, specs_file, out_dir = sys.argv[1:8]
    reps = int(reps)
    offsets = [int(x) for x in offsets.split()]
    split = json.loads((ROOT / "data/processed" / split_file).read_text())
    codes = split[set_name]
    imap = {c: i for i, c in enumerate(codes)}
    specs = json.loads(Path(specs_file).read_text())
    NMETH = len(specs)
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    alt = os.environ.get("RSSP_PYTHON")
    if alt:
        import multiprocessing as mp
        mp.set_executable(alt)
    import pickle
    inst = pickle.load(open(ROOT / "data/processed" / os.environ.get("RSSP_INSTANCES", "instances.pkl"), "rb"))
    size = {c: len(inst["pool"][c].dur) * inst["pool"][c].K for c in codes}
    tasks = []
    for code in sorted(codes, key=lambda c: -size[c]):
        have = done_reps(out / f"partial_{code}.csv")
        for off in offsets:
            for rep in range(off, off + reps):
                if rep not in have:
                    tasks.append((code, imap[code], seed_split, rep, specs, None, None, None, 0.0))
    print(len(tasks), "tasks to run", flush=True)
    t0 = time.time()
    writers = {}
    n = 0
    with Pool(processes=9, initializer=runner._init) as pool:
        for rows in pool.imap_unordered(runner._job, tasks, chunksize=1):
            if not rows:
                continue
            code = rows[0]["code"]
            path = out / f"partial_{code}.csv"
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if new:
                    w.writeheader()
                w.writerows(rows)
            n += 1
            if n % 100 == 0:
                with open(out / "progress.txt", "a") as g:
                    g.write(f"{n}/{len(tasks)} tasks in {time.time() - t0:.0f}s\n")
    with open(out / "progress.txt", "a") as g:
        g.write(f"finished {n} tasks in {time.time() - t0:.0f}s\n")
