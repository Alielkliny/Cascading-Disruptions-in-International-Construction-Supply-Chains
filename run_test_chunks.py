"""Remaining sealed-test blocks as per-project chunks (same random-number streams as whole blocks)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import runner
from run_stage import roster_specs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/runs/test_chunks"
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    offsets = [int(x) for x in sys.argv[1:]] or [180, 240]
    split = json.loads((ROOT / "data/processed/split.json").read_text())
    codes = split["test"]
    imap = {c: i for i, c in enumerate(codes)}
    for off in offsets:
        for code in codes:
            f = OUT / f"r{off}_{code}.csv"
            if f.exists():
                continue
            t0 = time.time()
            df = runner.run([code], "test", 60, roster_specs(), rep_offset=off, index_map=imap)
            df.to_csv(f, index=False)
            with open(ROOT / "results/runs/test_progress.txt", "a") as g:
                g.write(f"chunk {off} {code} done in {time.time()-t0:.0f}s\n")
