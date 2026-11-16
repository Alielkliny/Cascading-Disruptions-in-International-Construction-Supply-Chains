"""Experiments after the sealed test: case study, thresholds, compound shocks and sensitivity.

Usage: python experiments.py <name>
  case_trace   designed compound scenario on the Wind Farm case, every method, with traces
  case_mc      Monte Carlo on the Wind Farm case (case seeds), every method
  min_incentive  supplier best response over fine bonus and cost-share grids (planning model)
  timing       intervention delay sweep on the case (RSSP and practice)
  levers       lever choices of RSSP across route and supplier shock severity and duration
  bottleneck   shared route congestion (theta) and shared upstream supplier
  compound     single, paired and triple shocks: order, spacing and severity
  sensitivity  one-at-a-time sensitivity on test projects (fresh seeds)
"""
import itertools
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import core as cr  # noqa: E402
import engine as en  # noqa: E402
import runner  # noqa: E402
from run_stage import roster_specs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments"
OUT.mkdir(parents=True, exist_ok=True)
_I = None


def _init():
    global _I
    runner._init()
    _I = runner._INST


def case():
    return pickle.load(open(ROOT / "data/processed/instances.pkl", "rb"))["case"]


def designed_spec(inst):
    """Supplier outage at the start of turbine production, then route congestion and an approval backlog."""
    t1 = float(np.round(inst.rel[1]))
    return [[0.0, t1, 25.0, 0.8, 25.0],                 # supplier capacity to 20% for 25 days
            [1.0, t1 + 20.0, 30.0, 6.0 * 3.4736, 30.0],  # route congestion at the 2020 GSCPI peak
            [2.0, t1 + 45.0, 20.0, 0.75, 20.0]]          # approval capacity to 25% for 20 days


def case_trace():
    inst = case()
    spec = designed_spec(inst)
    rows, traces = [], {}
    for label, pname, params in roster_specs():
        sc = en.make_scenario(inst, 900001, spec=spec)
        r = en.simulate(inst, sc, en.Policy(pname, params), record=True)
        traces[label] = {"trace": r.pop("trace"), "approvals": r.pop("approvals")}
        rows.append({"method": label, **{k: v for k, v in r.items() if not isinstance(v, (list, dict))}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "case_trace_summary.csv", index=False)
    (OUT / "case_trace.json").write_text(json.dumps({"spec": spec, "rS": inst.rS.tolist(), "LSs": inst.LSs.tolist(),
                                                      "rel": inst.rel.tolist(), "plan_done": inst.plan_done.tolist(),
                                                      "traces": traces}, default=float))
    print(df[["method", "contractor_cost", "project_delay", "supplier_profit", "system_cost", "bonus_paid",
              "share_paid", "expedite", "alt_batches"]].round(0).to_string())


def case_mc(reps=300):
    df = runner.run(["CASE"], "case", reps, roster_specs())
    df.to_csv(OUT / "case_mc.csv", index=False)
    print(df.groupby("method")[["contractor_cost", "project_delay", "system_cost", "supplier_profit"]].mean().round(1))


def min_incentive():
    """Smallest bonus (share of batch value) or cost share at which the supplier's best response includes surge
    capacity, by supplier-shock severity and outage length, evaluated at the onset epoch on the case."""
    inst = case()
    E = en.GRID["E"] * inst.P[cr.P_EBAR]
    U = en.GRID["U"]
    ones = np.ones(len(inst.pcum))
    rows = []
    t1 = float(np.round(inst.rel[1]))
    for sev in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        for dur in [10.0, 20.0, 30.0]:
            S, Bt = en.init_state(inst)
            sh = np.array([[0.0, t1, dur, sev, dur]])
            epoch = t1 + 5.0 - (t1 % 5.0)
            cr.advance(S, Bt, inst.P, inst.rel, inst.K, epoch, sh, 1, ones, ones, ones, ones, 0, 0, 0.0, 0.0, 0.0, 0.0,
                       0.0, inst.pcum)
            nb = int(S[cr.NB])
            mask = np.array([1.0 if j >= nb else 0.0 for j in range(inst.K)])

            def br(phi, b):
                return en.follower_br(S, Bt, inst.P, inst.rel, inst.K, inst.rS, inst.LSs, inst.rLS, inst.sbLS,
                                      inst.rmLS, 0.0, 0.0, 0, phi, b * mask, 0, en.T_CAP, E, U, en.DSH, en.DNO,
                                      inst.pcum, 0, en.HEDGE["scales"], en.HEDGE["weights"])
            base = br(0.0, 0.0)
            b_min = next((b for b in np.round(np.arange(0.0, 0.401, 0.005), 3) if br(0.0, b)[1] > 0), np.nan)
            phi_min = next((p for p in np.round(np.arange(0.0, 1.001, 0.05), 2) if br(p, 0.0)[1] > 0), np.nan)
            e_b = br(0.0, b_min)[0] if b_min == b_min else np.nan
            rows.append({"severity": sev, "outage_days": dur, "base_effort": base[0], "base_surge": base[1],
                         "min_bonus_share": b_min, "min_cost_share": phi_min,
                         "min_bonus_eur": b_min * inst.wbar if b_min == b_min else np.nan, "effort_at_min_bonus": e_b,
                         "delay_no_incentive": base[5], "delay_at_min_bonus": br(0.0, b_min)[5] if b_min == b_min else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "min_incentive.csv", index=False)
    print(df.round(3).to_string())


def _rp():
    fd = ROOT / "results/design/final_design.json"
    return json.loads(fd.read_text()) if fd.exists() else {}


def _timing_job(args):
    import engine as en2
    d, rep = args
    inst = _I["case"]
    sc = en2.make_scenario(inst, 950000 + rep, spec=None)
    out = []
    for label in ("rssp", "practice"):
        r = en2.simulate(inst, sc, en2.Policy(label, _rp() if label == "rssp" else {}), intervention_delay_days=d)
        out.append({"delay_days": d, "rep": rep, "method": label, **{k: r[k] for k in runner.METRICS}})
    return out


def timing(reps=60):
    tasks = [(d, rep) for d in [0, 5, 10, 15, 20, 30, 40, 60] for rep in range(reps)]
    rows = []
    with Pool(9, initializer=_init) as p:
        for r in p.imap_unordered(_timing_job, tasks, chunksize=2):
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "timing.csv", index=False)
    print(df.groupby(["method", "delay_days"])[["contractor_cost", "project_delay"]].mean().round(1).to_string())


def _lever_job(args):
    import engine as en2
    typ, sev, dur, rep = args
    inst = _I["case"]
    t1 = float(np.round(inst.rel[1]))
    spec = [[float(typ), t1, float(dur), float(sev), float(dur)]]
    sc = en2.make_scenario(inst, 960000 + rep, spec=spec)
    out = []
    for label in ("rssp", "naive"):
        r = en2.simulate(inst, sc, en2.Policy(label, _rp() if label == "rssp" else {}), record=(label == "rssp"))
        rec = {"type": typ, "severity": sev, "duration": dur, "rep": rep, "method": label,
               **{k: r[k] for k in runner.METRICS}}
        if label == "rssp":
            tr = r["trace"]
            rec["weeks_bonus"] = sum(1 for x in tr if x["bonus_max"] > 0)
            rec["weeks_share"] = sum(1 for x in tr if x["phi"] > 0)
            rec["weeks_exp"] = sum(1 for x in tr if x["exp"] > 0)
            rec["weeks_rer"] = sum(1 for x in tr if x["rer"] > 0)
            rec["alt_used"] = float(r["alt_batches"] > 0)
        out.append(rec)
    return out


def levers(reps=20):
    tasks = []
    for sev in [0.5, 0.7, 0.9]:
        for dur in [10, 20, 30, 45]:
            for rep in range(reps):
                tasks.append((0, sev, dur, rep))
    for peak in [1.28, 1.81, 3.47, 4.43]:
        for dur in [10, 20, 30, 45]:
            for rep in range(reps):
                tasks.append((1, 6.0 * peak, dur, rep))
    rows = []
    with Pool(9, initializer=_init) as p:
        for r in p.imap_unordered(_lever_job, tasks, chunksize=2):
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "levers.csv", index=False)
    print(df[df.method == "rssp"].groupby(["type", "severity", "duration"])[["weeks_bonus", "weeks_share", "weeks_exp",
                                                                              "weeks_rer", "alt_used"]].mean().round(2).to_string())


def _bottleneck_job(args):
    import engine as en2
    from build_pool import CASE, CASE_LINKS
    theta, up, rep = args
    pr = pickle.load(open(ROOT / "data/processed/projects.pkl", "rb"))[CASE]
    inst = en2.build_instance(pr, cfg={"theta": theta, "upshared": up}, explicit_ids=CASE_LINKS)
    sc = en2.make_scenario(inst, 970000 + rep)
    out = []
    for label, pname, params in roster_specs():
        if label not in ("rssp", "nonstrategic_mpc", "trigger_rules", "practice", "naive", "centralized"):
            continue
        r = en2.simulate(inst, sc, en2.Policy(pname, params))
        out.append({"theta": theta, "upshared": up, "rep": rep, "method": label, **{k: r[k] for k in runner.METRICS}})
    return out


def bottleneck(reps=60):
    tasks = [(th, up, rep) for th in [0.0, 0.3, 0.6, 1.0] for up in [0.0, 1.0] for rep in range(reps)]
    rows = []
    with Pool(9, initializer=_init) as p:
        for r in p.imap_unordered(_bottleneck_job, tasks, chunksize=2):
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bottleneck.csv", index=False)
    print(df.groupby(["theta", "upshared", "method"])[["contractor_cost", "project_delay", "alt_batches",
                                                         "reroute"]].mean().round(1).to_string())


def _compound_job(args):
    import engine as en2
    key, spec, rep = args
    inst = _I["case"]
    sc = en2.make_scenario(inst, 980000 + rep, spec=spec if spec else np.zeros((0, 5)))
    out = []
    for label in ("naive", "rssp"):
        r = en2.simulate(inst, sc, en2.Policy(label, _rp() if label == "rssp" else {}))
        out.append({"key": key, "rep": rep, "method": label, **{k: r[k] for k in runner.METRICS}})
    return out


def compound(reps=40):
    inst = case()
    t1 = float(np.round(inst.rel[1]))
    shock = {0: (25.0, 0.8), 1: (30.0, 6.0 * 3.4736), 2: (20.0, 0.75)}
    tasks = []

    def mk(order, spacing, scale=1.0):
        spec, t = [], t1
        for i, ty in enumerate(order):
            if i > 0:
                t += spacing
            dur, sev = shock[ty]
            sev = min(sev * scale, 0.95) if ty != 1 else sev * scale
            spec.append([float(ty), t, dur, sev, dur])
        return spec

    for rep in range(reps):
        tasks.append(("none", [], rep))
        for ty in range(3):
            tasks.append((f"single_{ty}", mk([ty], 0), rep))
        for order in itertools.permutations(range(3), 2):
            for g in [5, 20, 40, 60]:
                tasks.append((f"pair_{order[0]}{order[1]}_g{g}", mk(order, g), rep))
        for order in itertools.permutations(range(3)):
            for g in [5, 20, 40]:
                tasks.append((f"triple_{''.join(map(str, order))}_g{g}", mk(order, g), rep))
        for scale in [0.6, 0.8, 1.2]:
            tasks.append((f"triple_012_g20_s{scale}", mk((0, 1, 2), 20, scale), rep))
            for ty in range(3):
                tasks.append((f"single_{ty}_s{scale}", mk([ty], 0, scale), rep))
    rows = []
    with Pool(9, initializer=_init) as p:
        for r in p.imap_unordered(_compound_job, tasks, chunksize=4):
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "compound.csv", index=False)
    print(df.groupby(["key", "method"])[["project_delay", "contractor_cost"]].mean().round(1).head(60).to_string())


SENS = {
    "cd_share": (0.0005, 0.002), "pen_d": (0.0005, 0.002), "e_bar": (0.01, 0.04), "rho0": (0.08, 0.25),
    "rho1": (0.4, 0.8), "h_u": (0.15, 0.4), "cost_exp": (0.03, 0.12), "pi_alt": (0.1, 0.3), "Q0": (25.0, 60.0),
    "T0": (18.0, 35.0), "theta": (0.0, 0.7), "alpha_route": (3.0, 9.0), "est_sigma": (0.15, 0.5),
    "phi_w": (0.9056, 0.9869), "prod_days_per_batch": (7.0, 14.0), "link_rule": ("cost", "cost"),
}


def sensitivity(reps=12):
    split = json.loads((ROOT / "data/processed/split2.json").read_text())
    base = [s for s in roster_specs() if s[0] in ("rssp", "nonstrategic_mpc", "static_share", "practice", "naive",
                                                  "trigger_rules")]
    with_ablation = base + [s for s in roster_specs() if s[0] == "rssp_memoryless"]
    persistence = ("phi_w", "rho0", "rho1", "est_sigma")
    frames = []
    done = {p.stem for p in OUT.glob("sens_*.csv")}
    for par, (lo, hi) in SENS.items():
        for tag, val in (("low", lo), ("high", hi)):
            if par == "link_rule" and tag == "high":
                continue
            name = f"sens_{par}_{tag}"
            if name in done:
                frames.append(pd.read_csv(OUT / f"{name}.csv"))
                continue
            t0 = time.time()
            specs = with_ablation if par in persistence else base
            df = runner.run(split["test"], "sens2", reps, specs, cfg={par: val})
            df["param"], df["level"], df["value"] = par, tag, val
            df.to_csv(OUT / f"{name}.csv", index=False)
            frames.append(df)
            print(name, f"{time.time() - t0:.0f}s", flush=True)
    pd.concat(frames).to_csv(OUT / "sensitivity_all.csv", index=False)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
