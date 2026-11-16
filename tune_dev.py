"""Tune design parameters on development projects (equal budget per method)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import runner
ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    tag = sys.argv[1]; reps = int(sys.argv[2])
    specs = json.loads(Path(sys.argv[3][1:]).read_text()) if sys.argv[3].startswith("@") else json.loads(sys.argv[3])
    cfg = json.loads(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
    grid = json.loads(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    split = json.loads((ROOT / "data/processed/split.json").read_text())
    t0 = time.time()
    df = runner.run(split["development"], "development", reps, specs, cfg=cfg, grid=grid)
    df["cfg"] = json.dumps(cfg); df["grid"] = json.dumps(grid)
    out = ROOT / f"results/design/{tag}.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    piv = df.groupby(["code", "method"])["contractor_cost"].mean().unstack()
    print(f"{time.time()-t0:.0f}s")
    print(piv.round(0).to_string())
    print(df.groupby("method")[["contractor_cost","project_delay","system_cost"]].mean().round(1).to_string())
