"""Per-project chunked runs with the random-number stream of the full split (restartable).

usage: python run_chunks.py SPLIT_FILE SET SEED_SPLIT REPS OFFSET SPECS_JSON OUT_DIR
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import runner  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    split_file, set_name, seed_split, reps, offset, specs_file, out_dir = sys.argv[1:8]
    reps, offset = int(reps), int(offset)
    split = json.loads((ROOT / "data/processed" / split_file).read_text())
    codes = split[set_name]
    imap = {c: i for i, c in enumerate(codes)}
    specs = json.loads(Path(specs_file).read_text())
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    for code in codes:
        f = out / f"r{offset}_{code}.csv"
        if f.exists():
            continue
        t0 = time.time()
        df = runner.run([code], seed_split, reps, specs, rep_offset=offset, index_map=imap)
        df.to_csv(f, index=False)
        with open(out / "progress.txt", "a") as g:
            g.write(f"{offset} {code} {time.time() - t0:.0f}s\n")
