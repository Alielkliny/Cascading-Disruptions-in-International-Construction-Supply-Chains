"""Build the instance pool from admitted DSLIB projects and split it by project into development and test sets."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import engine as en

ROOT = Path(__file__).resolve().parents[1]
CASE = "C2011-13"
CASE_LINKS = [94, 95, 96, 97, 98, 99, 100]  # turbine (WEC) erection activities of the Wind Farm


def main():
    P = pickle.load(open(ROOT / "data/processed/projects.pkl", "rb"))
    rows = []
    pool = {}
    for code, pr in sorted(P.items()):
        if code == CASE:
            continue
        inst = en.build_instance(pr)
        ok = inst is not None
        rows.append({"code": code, "name": pr.name, "n_act": len(pr.dur), "pd": float(pr.pd_sheet), "built": ok,
                     "K": inst.K if ok else None,
                     "links": inst.meta["link_names"] if ok else None,
                     "wbar": inst.wbar if ok else None, "cd": inst.cd if ok else None})
        if ok:
            pool[code] = inst
    # near-duplicate networks (same activity count, duration and linked activities) are represented once
    seen, dups = {}, []
    for code in sorted(pool):
        r = next(x for x in rows if x["code"] == code)
        sig = (r["n_act"], r["pd"], tuple(r["links"]))
        if sig in seen:
            dups.append(code)
        else:
            seen[sig] = code
    for code in dups:
        pool.pop(code)
    codes = sorted(pool)
    rng = np.random.default_rng(20261102)
    perm = list(rng.permutation(codes))
    n_dev = int(round(0.4 * len(codes)))
    dev, test = sorted(perm[:n_dev]), sorted(perm[n_dev:])
    case = en.build_instance(P[CASE], explicit_ids=CASE_LINKS)
    out = {"case": CASE, "case_links": CASE_LINKS, "pool": codes, "development": dev, "test": test,
           "split_seed": 20261102, "dropped_duplicates": dups, "rule": "admitted DSLIB construction projects with K>=4 linked activities "
           "(top-6 fixed-cost activities with baseline start >= 80 working days) and a non-negative production release",
           "rows": rows}
    (ROOT / "data/processed/split.json").write_text(json.dumps(out, indent=2, default=str))
    with open(ROOT / "data/processed/instances.pkl", "wb") as f:
        pickle.dump({"pool": pool, "case": case}, f)
    print(len(codes), "pool;", len(dev), "dev;", len(test), "test")
    for r in rows:
        print(r["code"], r["name"][:32], r["n_act"], r["pd"], r["built"], r["K"], (r["links"] or [])[:3])


if __name__ == "__main__":
    main()
