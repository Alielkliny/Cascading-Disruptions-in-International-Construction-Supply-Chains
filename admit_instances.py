"""Admit DSLIB construction projects as instances: parse, CPM check against PD, tracking check.

Writes data/processed/instances.json, data/processed/admission.csv and results/schedule_validation.json.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dslib import load_project, cpm_forward  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
XL = ROOT / "data/raw/dslib/extracted/DSLIB 3.4"
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    sheet = pd.read_excel(XL / "DSLIB_Analysis_Sheet.xlsx", "DSLIB", header=2)
    sheet = sheet[sheet["Code"].astype(str).str.match(r"^C\d{4}-\d+")].copy()
    n_all = len(sheet)
    cons = sheet[sheet["Sector"].astype(str).str.contains("Construction", case=False)].copy()
    n_cons = len(cons)
    tracked = cons[cons["Tracking"].isin(["G", "Y"])].copy()
    n_tracked = len(tracked)
    files = {p.name.split(" ")[0]: p for p in (XL / "Excel").glob("*.xlsx")}
    rows, projects = [], {}
    for _, r in tracked.iterrows():
        code = r["Code"]
        rec = {"code": code, "name": r["Project name"], "sector": r["Sector"], "n_sheet": r["# activities"], "pd_sheet": r["PD (days)"]}
        p = files.get(code)
        if p is None:
            rec.update(status="no_file")
            rows.append(rec)
            continue
        try:
            pr = load_project(str(p), r.to_dict())
        except Exception as exc:  # parse failure is recorded, never repaired by hand
            rec.update(status=f"parse_error: {exc}")
            rows.append(rec)
            continue
        release = np.nan_to_num(pr.es_sheet, nan=0.0)
        es, ef = cpm_forward(pr.dur, pr.preds, pr.order, release=release)
        pd_cpm = float(ef.max())
        # baseline precedence feasibility: share of activities whose sheet start already satisfies its relations
        feas = float(np.mean(np.abs(es - release) <= 0.51))
        rec.update(n_leaf=len(pr.dur), pd_cpm=pd_cpm, baseline_feasible_share=feas, sha256=sha256(p),
                   hours_per_day=pr.meta.get("hours_per_day"), n_relations=int(sum(len(x) for x in pr.preds)))
        rel_err = abs(pd_cpm - pr.pd_sheet) / max(pr.pd_sheet, 1.0)
        rec["pd_rel_err"] = rel_err
        # observed completion from tracking: max(actual start + actual duration)
        if pr.actual_dur is not None and np.isfinite(pr.actual_dur).sum() > 0 and np.isfinite(pr.actual_start).sum() > 0:
            ok = np.isfinite(pr.actual_dur) & np.isfinite(pr.actual_start)
            rd_obs = float(np.nanmax(pr.actual_start[ok] + pr.actual_dur[ok]))
            ad = np.where(np.isfinite(pr.actual_dur), pr.actual_dur, pr.dur)
            _, ef_a = cpm_forward(ad, pr.preds, pr.order, release=release)
            rec.update(rd_obs=rd_obs, rd_cpm_actual=float(ef_a.max()), share_tracked=float(ok.mean()))
        status = "admitted"
        if len(pr.dur) < 15:
            status = "too_small"
        elif pr.pd_sheet < 60:
            status = "too_short"
        elif rel_err > 0.02:
            status = "cpm_mismatch"
        elif feas < 0.95:
            status = "baseline_infeasible"
        elif int(sum(len(x) for x in pr.preds)) < len(pr.dur) * 0.5:
            status = "sparse_network"
        rec["status"] = status
        rows.append(rec)
        if status == "admitted":
            projects[code] = pr
    adm = pd.DataFrame(rows)
    adm.to_csv(OUT / "admission.csv", index=False)
    with open(OUT / "projects.pkl", "wb") as f:
        pickle.dump(projects, f)
    attrition = {
        "dslib_projects": int(n_all), "construction": int(n_cons), "construction_tracked": int(n_tracked),
        "status_counts": adm["status"].value_counts().to_dict(), "admitted": sorted(projects),
        "zip_sha256": sha256(ROOT / "data/raw/dslib/DSLIB3.4.zip"),
    }
    (OUT / "admission_summary.json").write_text(json.dumps(attrition, indent=2, default=str))
    print(json.dumps(attrition, indent=2, default=str))
    print(adm[["code", "name", "n_leaf", "pd_sheet", "pd_cpm", "pd_rel_err", "rd_obs", "rd_cpm_actual", "status"]].to_string())


if __name__ == "__main__":
    main()
