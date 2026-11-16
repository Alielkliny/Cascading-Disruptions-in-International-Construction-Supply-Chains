"""Open the sealed test once: blocks of 60 replications per project until 600, each block saved as it completes."""
import subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
for off in range(0, 600, 60):
    out = ROOT / f"results/runs/test_test_r{off}_60.csv"
    if out.exists():
        continue
    t0 = time.time()
    subprocess.run([PY, "-P", "run_stage.py", "test", "test", "60", str(off)], cwd=str(Path(__file__).parent), check=True)
    with open(ROOT / "results/runs/test_progress.txt", "a") as f:
        f.write(f"block {off} done in {time.time()-t0:.0f}s\n")
