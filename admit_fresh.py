"""Fresh held-out pool: DSLIB construction projects WITHOUT tracking data, admitted by the unchanged rules.

Same parser, same admission thresholds, same package rule and a duplicate check against every existing instance.
Writes data/processed/fresh_admission.csv, data/processed/fresh_instances.pkl and data/processed/fresh_split.json.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dslib import load_project, cpm_forward  # noqa: E402
import engine as en  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
XL = ROOT / "data/raw/dslib/extracted/DSLIB 3.4"


def main():
    sheet = pd.read_excel(XL / "DSLIB_Analysis_Sheet.xlsx", "DSLIB", header=2)
    sheet = sheet[sheet["Code"].astype(str).str.match(r"^C\d{4}-\d+")].copy()
    cons = sheet[sheet["Sector"].astype(str).str.contains("Construction", case=False)].copy()
    fresh = cons[~cons["Tracking"].isin(["G", "Y"])].copy()
    files = {p.name.split(" ")[0]: p for p in (XL / "Excel").glob("*.xlsx")}
    rows, projects = [], {}
    for _, r in fresh.iterrows():
        code = r["Code"]
        rec = {"code": code, "name": r["Project name"], "tracking": r["Tracking"]}
        p = files.get(code)
        if p is None:
            rec["status"] = "no_file"
            rows.append(rec)
            continue
        try:
            pr = load_project(str(p), r.to_dict(), with_tracking=False)
        except Exception as exc:
            rec["status"] = f"parse_error: {exc}"
            rows.append(rec)
            continue
        release = np.nan_to_num(pr.es_sheet, nan=0.0)
        es, ef = cpm_forward(pr.dur, pr.preds, pr.order, release=release)
        pd_cpm = float(ef.max())
        feas = float(np.mean(np.abs(es - release) <= 0.51))
        rel_err = abs(pd_cpm - pr.pd_sheet) / max(pr.pd_sheet, 1.0) if np.isfinite(pr.pd_sheet) else np.inf
        nrel = int(sum(len(x) for x in pr.preds))
        status = "admitted"
        if len(pr.dur) < 15:
            status = "too_small"
        elif not np.isfinite(pr.pd_sheet) or pr.pd_sheet < 60:
            status = "too_short"
        elif rel_err > 0.02:
            status = "cpm_mismatch"
        elif feas < 0.95:
            status = "baseline_infeasible"
        elif nrel < len(pr.dur) * 0.5:
            status = "sparse_network"
        rec.update(n_leaf=len(pr.dur), pd_sheet=pr.pd_sheet, pd_cpm=pd_cpm, rel_err=rel_err, feas=feas, status=status)
        rows.append(rec)
        if status == "admitted":
            projects[code] = pr
    adm = pd.DataFrame(rows)
    adm.to_csv(ROOT / "data/processed/fresh_admission.csv", index=False)
    # existing instances for the duplicate check
    old = pickle.load(open(ROOT / "data/processed/instances.pkl", "rb"))
    old_split = json.loads((ROOT / "data/processed/split.json").read_text())
    old_sigs = set()
    for r in old_split["rows"]:
        if r["built"]:
            old_sigs.add((r["n_act"], float(r["pd"]), tuple(r["links"])))
    case = old["case"]
    pool, built_rows, dups = {}, [], []
    for code, pr in sorted(projects.items()):
        inst = en.build_instance(pr)
        if inst is None:
            built_rows.append({"code": code, "built": False})
            continue
        sig = (len(pr.dur), float(pr.pd_sheet), tuple(inst.meta["link_names"]))
        if sig in old_sigs or any(sig == s2 for s2 in [(len(p2.dur), float(p2.pd_sheet), tuple(i2.meta["link_names"]))
                                                       for c2, (p2, i2) in pool.items()]):
            dups.append(code)
            continue
        pool[code] = (pr, inst)
        built_rows.append({"code": code, "built": True, "K": inst.K, "n_act": len(pr.dur), "pd": pr.pd_sheet,
                           "links": inst.meta["link_names"]})
    summary = {"construction_untracked": int(len(fresh)), "status_counts": adm["status"].value_counts().to_dict(),
               "admitted": len(projects), "built": len(pool), "duplicates_dropped": dups,
               "fresh_test": sorted(pool), "rows": built_rows}
    (ROOT / "data/processed/fresh_split.json").write_text(json.dumps(summary, indent=2, default=str))
    with open(ROOT / "data/processed/fresh_instances.pkl", "wb") as f:
        pickle.dump({c: i for c, (p, i) in pool.items()}, f)
    with open(ROOT / "data/processed/fresh_projects.pkl", "wb") as f:
        pickle.dump({c: p for c, (p, i) in pool.items()}, f)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1, default=str))


if __name__ == "__main__":
    main()
