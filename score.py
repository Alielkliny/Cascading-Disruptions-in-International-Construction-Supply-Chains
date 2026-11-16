"""Score a results table: normalized metrics, paired tests with Holm adjustment, comparison_scores rows."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parents[1]
PROPOSED = "rssp"
ROSTER = ["naive", "practice", "static_share", "static_bonus", "static_two_part", "trigger_rules", "nonstrategic_mpc",
          "rssp_memoryless"]
TIE_TOL = 1e-9  # days or working days; declared in protocol Amendment 4
WIN_METRICS = {"contractor_cost_days": "min", "project_delay": "min", "system_cost_days": "min"}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    inst = pickle.load(open(ROOT / "data/processed/instances_v2.pkl", "rb"))
    cd = {c: i.cd for c, i in inst["pool"].items()}
    cd["CASE"] = inst["case"].cd
    df = df.copy()
    df["cd"] = df["code"].map(cd)
    for m in ["contractor_cost", "system_cost", "supplier_profit", "intervention_spend", "transfers", "bonus_paid",
              "share_paid", "penalty", "wait_cost", "delay_cost", "expedite", "reroute", "effort", "surge", "alt_premium"]:
        df[m + "_days"] = df[m] / df["cd"]
    return df


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        run = max(run, adj)
        out[k] = run
    return out


def paired_tests(df: pd.DataFrame, metric: str, proposed=PROPOSED, roster=ROSTER):
    """Replication-level paired Wilcoxon signed-rank (common random numbers pair each project and scenario)."""
    piv = df.pivot_table(index=["code", "rep"], columns="method", values=metric)
    raw = {}
    for m in roster:
        if m not in piv.columns:
            continue
        d = (piv[proposed] - piv[m]).dropna()
        d = d.where(d.abs() > TIE_TOL, 0.0)  # storage round-off (about 1e-15 days) is a tie, not a difference
        if np.allclose(d, 0):
            raw[m] = 1.0
        else:
            raw[m] = float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    return raw, holm(raw)


def project_level(df: pd.DataFrame, metric: str, proposed=PROPOSED, roster=ROSTER):
    piv = df.groupby(["code", "method"])[metric].mean().unstack()
    out = {}
    for m in roster:
        if m not in piv.columns:
            continue
        d = piv[proposed] - piv[m]
        tol = 1e-9
        nz = d[d.abs() > tol]
        p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 2 and not np.allclose(nz, 0) else 1.0
        out[m] = {"wins": int((d < -tol).sum()), "ties": int((d.abs() <= tol).sum()), "losses": int((d > tol).sum()),
                  "mean_diff": float(d.mean()), "p_wilcoxon_projects": p}
    return out


def score(df: pd.DataFrame, stage: str, split: str, idea="A"):
    df = normalize(df)
    rows, summary = [], {"stage": stage, "split": split, "n_projects": int(df["code"].nunique()),
                         "n_reps": int(df["rep"].nunique()), "metrics": {}}
    for metric in WIN_METRICS:
        means = df.groupby("method")[metric].mean()
        raw, adj = paired_tests(df, metric)
        summary["metrics"][metric] = {"means": means.to_dict(), "p_raw": raw, "p_holm": adj,
                                      "project_level": project_level(df, metric)}
        for m in [PROPOSED] + ROSTER:
            if m not in means.index:
                continue
            rows.append({"idea": idea, "stage": stage, "split": split, "method": m, "metric": metric,
                         "value": float(means[m]), "p_holm": "" if (m == PROPOSED or stage == "probe") else adj.get(m, "")})
    return pd.DataFrame(rows), summary


def precision(df: pd.DataFrame, metric="contractor_cost_days"):
    df = normalize(df)
    out = {}
    for m, g in df.groupby("method"):
        per = g.groupby("code")[metric].agg(["mean", "var", "count"])
        P = len(per)
        mean = float(per["mean"].mean())
        half = float(1.96 * np.sqrt((per["var"] / per["count"]).sum() / P ** 2))
        out[m] = {"n_per_project": int(per["count"].min()), "projects": P, "mean": mean, "half_width_strat": half,
                  "rel_half_width_strat": half / abs(mean) if mean != 0 else None}
    return out


def append_scores(new: pd.DataFrame, path=ROOT / "comparison_scores.csv"):
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        key = ["idea", "stage", "split", "method", "metric"]
        new = new.astype(str)
        merged = pd.concat([old[~old.set_index(key).index.isin(new.set_index(key).index)], new])
    else:
        merged = new
    merged.to_csv(path, index=False)


if __name__ == "__main__":
    src, stage, split = sys.argv[1], sys.argv[2], sys.argv[3]
    idea = sys.argv[4] if len(sys.argv) > 4 else "A"
    df = pd.read_csv(src)
    rows, summ = score(df, stage, split, idea=idea)
    append_scores(rows)
    summ["precision"] = precision(df)
    out = ROOT / f"results/score_{stage}_{split}{'' if idea == 'A' else '_' + idea}.json"
    out.write_text(json.dumps(summ, indent=2, default=float))
    for metric, v in summ["metrics"].items():
        print(metric)
        for m, mean in sorted(v["means"].items(), key=lambda kv: kv[1]):
            pl = v["project_level"].get(m, {})
            print(f"   {m:18s} {mean:9.3f}  p_holm={v['p_holm'].get(m, float('nan')):.3g}  W/T/L={pl.get('wins')}/{pl.get('ties')}/{pl.get('losses')}")
