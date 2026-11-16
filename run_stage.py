"""Run the full roster on one split and stage (probe, development or test) with tuned comparator settings."""
import json, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import runner

ROOT = Path(__file__).resolve().parents[1]


def roster_specs():
    tuned = json.loads((ROOT / "results/design/tuned_comparators.json").read_text())["tuned"]
    fd = ROOT / "results/design/final_design.json"
    rp = json.loads(fd.read_text()) if fd.exists() else {}
    specs = [["rssp", "rssp", dict(rp)], ["rssp_memoryless", "rssp_memoryless", dict(rp)],
             ["nonstrategic_mpc", "nonstrategic_mpc", {}], ["naive", "naive", {}], ["practice", "practice", {}],
             ["centralized", "centralized", {}]]
    for fam, label in tuned.items():
        params = {}
        for kv in label.split("|")[1:]:
            k, v = kv.split("=")
            key = {"phi": "phi", "b": "bonus", "m": "mode", "te": "th_exp", "tr": "th_rer", "ta": "th_alt"}[k]
            params[key] = float(v)
        specs.append([fam, fam, params])
    return specs


if __name__ == "__main__":
    stage, split_name, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rep_offset = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    split = json.loads((ROOT / "data/processed/split.json").read_text())
    codes = ["CASE"] if split_name == "case" else split[split_name]
    seed_split = {"probe": "probe", "development": "development", "test": "test", "case": "case"}[stage if stage != "development" else "development"]
    t0 = time.time()
    df = runner.run(codes, seed_split, reps, roster_specs(), rep_offset=rep_offset)
    out = ROOT / f"results/runs/{stage}_{split_name}_r{rep_offset}_{reps}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{time.time() - t0:.0f}s rows={len(df)} -> {out}")
