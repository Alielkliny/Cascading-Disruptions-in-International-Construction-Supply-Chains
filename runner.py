"""Parallel experiment runner with common random numbers across policies."""
from __future__ import annotations

import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
_INST = None
SEED_BASE = {"development": 10_000, "test": 500_000, "case": 900_000, "probe": 50_000, "sens": 700_000,
             "development2": 20_000, "test2": 600_000, "sens2": 750_000}
METRICS = ["contractor_cost", "project_delay", "system_cost", "supplier_profit", "recovery_days", "intervention_spend",
           "transfers", "bonus_paid", "share_paid", "penalty", "effort", "surge", "expedite", "reroute", "alt_batches",
           "wait_cost", "delay_cost", "milestones_on_time", "alt_premium"]


def _init():
    global _INST
    if os.environ.get("RSSP_LOW_PRIORITY"):  # opt-in only: below-normal workers starve when the machine is shared
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            pass
    with open(ROOT / "data/processed" / os.environ.get("RSSP_INSTANCES", "instances.pkl"), "rb") as f:
        _INST = pickle.load(f)


def seed_for(split, inst_index, rep):
    return SEED_BASE[split] + inst_index * 1000 + rep


def _job(args):
    import engine as en
    code, inst_index, split, rep, specs, cfg, grid, scen_kw, delay_days = args
    if grid:
        for k, v in grid.items():
            en.GRID[k] = np.array(v)
    inst = _INST["case"] if code == "CASE" else _INST["pool"][code]
    if cfg:
        import pickle as _p
        pr_all = _p.load(open(ROOT / "data/processed" / os.environ.get("RSSP_PROJECTS", "projects.pkl"), "rb"))
        from build_pool import CASE, CASE_LINKS
        pr = pr_all[CASE if code == "CASE" else code]
        inst = en.build_instance(pr, cfg=cfg, explicit_ids=CASE_LINKS if code == "CASE" else None)
        if inst is None:
            return []
    seed = seed_for(split, inst_index, rep)
    sc = en.make_scenario(inst, seed, cfg=cfg, **(scen_kw or {}))
    rows = []
    for label, pname, params in specs:
        t0 = time.time()
        r = en.simulate(inst, sc, en.Policy(pname, params), intervention_delay_days=delay_days)
        row = {"split": split, "code": inst.code if code != "CASE" else "CASE", "rep": rep, "seed": seed,
               "method": label, "secs": time.time() - t0, "n_shocks": int(sc.shocks.shape[0]),
               "shock_types": "".join(str(int(x)) for x in sc.shocks[:, 0])}
        for m in METRICS:
            row[m] = r[m]
        rows.append(row)
    return rows


def run(codes, split, reps, specs, cfg=None, grid=None, scen_kw=None, workers=9, delay_days=0.0, rep_offset=0,
        index_map=None):
    """codes: list of instance codes (or ['CASE']); returns a DataFrame with one row per (instance, rep, method).

    index_map: optional {code: index} so a subset of a split keeps the random-number stream of the full split."""
    import os as _os
    alt = _os.environ.get("RSSP_PYTHON")
    if alt:
        import multiprocessing as _mp
        _mp.set_executable(alt)
    tasks = []
    for i, code in enumerate(codes):
        idx = 999 if code == "CASE" else (index_map[code] if index_map else i)
        for rep in range(rep_offset, rep_offset + reps):
            tasks.append((code, idx, split, rep, specs, cfg, grid, scen_kw, delay_days))
    out = []
    with Pool(processes=min(workers, len(tasks)), initializer=_init) as pool:
        for rows in pool.imap_unordered(_job, tasks, chunksize=1):
            out.extend(rows)
    return pd.DataFrame(out)


def instance_index(code, split_codes):
    return split_codes.index(code)
