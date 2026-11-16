"""Apply the Amendment 1 selection rule to the development run of the delay-premium grid."""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "results/design/select_k/r*_*.csv"))])
    df = score.normalize(df)
    df.to_csv(ROOT / "results/design/select_k_all.csv", index=False)
    means = df.groupby("method")[["contractor_cost_days", "project_delay", "system_cost_days"]].mean()
    comps = [m for m in means.index if not m.startswith("rssp") and m != "centralized"]
    rows = []
    for k in ("0.0", "0.5", "1.0", "2.0"):
        m = f"rssp_k{k}"
        c, d = means.loc[m, "contractor_cost_days"], means.loc[m, "project_delay"]
        lead = min((means.loc[x, "contractor_cost_days"] - c) / means.loc[x, "contractor_cost_days"] for x in comps)
        gap = max(d - means.loc[x, "project_delay"] for x in comps)
        rows.append({"kappa": float(k), "cost": c, "delay": d, "min_cost_lead": lead, "max_delay_gap": gap,
                     "meets_main": gap <= 0.5 and lead >= 0.10, "within_1day": gap <= 1.0})
    t = pd.DataFrame(rows)
    main = t[t.meets_main]
    if len(main):
        chosen = float(main.kappa.min()); rule = "smallest kappa meeting both conditions"
    elif t.within_1day.any():
        chosen = float(t[t.within_1day].sort_values("min_cost_lead").iloc[-1].kappa); rule = "largest minimum cost lead within 1.0 day"
    else:
        chosen = None; rule = "none within 1.0 day: report it"
    out = {"table": t.to_dict(orient="records"), "chosen_kappa": chosen, "rule_applied": rule,
           "comparator_means": means.to_dict(orient="index")}
    (ROOT / "results/design/select_kappa.json").write_text(json.dumps(out, indent=1, default=float))
    print(means.round(3).to_string())
    print(t.round(3).to_string())
    print("chosen", chosen, rule)
