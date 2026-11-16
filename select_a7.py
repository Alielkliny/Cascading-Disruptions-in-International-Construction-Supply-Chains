"""Apply the Amendment 7 selection rule: 26 development projects, 120 replications."""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["k0", "k05_d3", "k1_d5", "k2_d5", "k2_d8"]
COMPS = ["naive", "practice", "static_share", "static_bonus", "static_two_part", "trigger_rules", "nonstrategic_mpc"]
GUARD = 0.6


def load(pattern):
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / pattern))])
    return score.normalize(df.drop_duplicates(["code", "rep", "method"], keep="last"))


if __name__ == "__main__":
    base = pd.read_csv(ROOT / "results/design/select_k_all.csv")
    base = base[base.method.isin(COMPS + ["centralized"])]
    comp = load("results/design/a7_comp/partial_*.csv")
    cand = load("results/design/a7/partial_*.csv")
    cols = list(base.columns)
    allrows = pd.concat([base, comp[cols], cand[cols]])
    allrows.to_csv(ROOT / "results/design/a7_all.csv", index=False)
    rows = []
    for v in VARIANTS:
        sub = allrows[allrows.method.isin(COMPS + [f"rssp_{v}", f"ml_{v}"])].copy()
        sub["method"] = sub["method"].replace({f"rssp_{v}": "rssp", f"ml_{v}": "rssp_memoryless"})
        cnt = sub.groupby(["code", "rep"]).method.nunique()
        full = cnt[cnt == len(COMPS) + 2].index
        sub = sub.set_index(["code", "rep"]).loc[full].reset_index()
        means = sub.groupby("method")[["contractor_cost_days", "project_delay", "system_cost_days"]].mean()
        raw, adj = score.paired_tests(sub, "contractor_cost_days")
        c, d = means.loc["rssp", "contractor_cost_days"], means.loc["rssp", "project_delay"]
        leads = {m: (means.loc[m, "contractor_cost_days"] - c) / means.loc[m, "contractor_cost_days"]
                 for m in COMPS + ["rssp_memoryless"]}
        gaps = {m: d - means.loc[m, "project_delay"] for m in COMPS}
        pl = score.project_level(sub, "contractor_cost_days")
        ok = (all(leads[m] >= 0.05 and adj[m] < 0.05 for m in COMPS) and max(gaps.values()) <= GUARD
              and leads["rssp_memoryless"] >= 0.05)
        rows.append({"variant": v, "pairs": len(full), "cost": c, "delay": d,
                     "system": means.loc["rssp", "system_cost_days"],
                     "min_lead": min(leads[m] for m in COMPS), "lead_vs_ablation": leads["rssp_memoryless"],
                     "max_holm_p": max(adj[m] for m in COMPS), "max_delay_gap": max(gaps.values()),
                     "lead_mpc": leads["nonstrategic_mpc"], "p_mpc": adj["nonstrategic_mpc"],
                     "wtl_mpc": "{wins}/{ties}/{losses}".format(**pl["nonstrategic_mpc"]),
                     "gap_two_part": gaps["static_two_part"], "gap_share": gaps["static_share"], "qualifies": ok})
    t = pd.DataFrame(rows)
    q = t[t.qualifies]
    chosen = None if q.empty else q.sort_values("min_lead").iloc[-1].variant
    out = {"rule": "Amendment 7", "table": t.to_dict(orient="records"), "chosen": chosen,
           "comparator_means": allrows[allrows.method.isin(COMPS + ["centralized"])].groupby("method")[
               ["contractor_cost_days", "project_delay", "system_cost_days"]].mean().to_dict(orient="index")}
    (ROOT / "results/design/select_a7.json").write_text(json.dumps(out, indent=1, default=float))
    pd.set_option("display.width", 250)
    print(t.round(4).to_string())
    print("chosen", chosen)
