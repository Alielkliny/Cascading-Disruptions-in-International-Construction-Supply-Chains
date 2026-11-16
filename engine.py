"""Instances, disruption scenarios, policies and the discrete-event simulation driver."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit

import core as cr
from core import advance, approvals_sorted, plan_payoffs, NS, NBF, NP
from dslib import cpm_backward, cpm_forward
from solvers import follower_br, leader_solve, central_solve
from fastplan import build_cache, leader_fast, central_fast

USE_FAST = True  # exact cached search (fastplan); False runs the direct search for equivalence checks

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------------------
# Parameter register (provenance tags in params_register.csv)
# ------------------------------------------------------------------------------------------
DEFAULTS = {
    "prod_days_per_batch": 10.0,   # ASSUMPTION
    "alt_prod_days_per_batch": 10.0,  # ASSUMPTION
    "inland": 3.0,                 # ASSUMPTION
    "vessel": 5.0,                 # ASSUMPTION weekly service
    "T0": 25.0,                    # ASSUMPTION ocean transit, working days
    "C0": 4.0,                     # ASSUMPTION clearance
    "A0": 4.0,                     # ASSUMPTION inspection
    "a_rev": 1.0,                  # ASSUMPTION review capacity per day
    "dR": 8.0,                     # ASSUMPTION reroute extra transit
    "theta": 0.3,                  # ASSUMPTION share of route congestion felt on the alternative route
    "fe": 0.5,                     # ASSUMPTION expedited transit factor
    "cost_exp": 0.06,              # ASSUMPTION share of batch value per expedited batch
    "cost_rer": 0.02,              # ASSUMPTION share of batch value per rerouted batch
    "pi_alt": 0.20,                # ASSUMPTION alternative-source price premium
    "f_c": 0.05,                   # ASSUMPTION reallocation compensation to the primary supplier
    "Q0": 40.0,                    # ASSUMPTION qualification and technical approval of an alternative source
    "margin": 0.15,                # ASSUMPTION supplier margin
    "pen_d": 0.001,                # ASSUMPTION late-delivery damages per day, share of batch value
    "pen_cap": 0.10,               # ASSUMPTION cap on late-delivery damages
    "e_bar": 0.02,                 # ASSUMPTION effort scale, share of batch value per week
    "rho0": 0.15,                  # ASSUMPTION natural weekly recovery share
    "rho1": 0.60,                  # ASSUMPTION maximum weekly recovery share
    "h_u": 0.25,                   # ASSUMPTION surge cost per surge batch, share of batch value
    "phi_w": None,                 # FITTED from GSCPI (weekly persistence)
    "cd_share": 0.001,             # ASSUMPTION daily delay cost, share of BAC
    "buf": 10.0,                   # ASSUMPTION contractual delivery buffer before need date
    "upshared": 0.0,               # scenario flag: alternative source shares the upstream supplier
    "m_days": 20.0,                # design: commitment block of the planning model (4 weeks)
    "remob_days": 2.0,             # ASSUMPTION resequencing (demobilize and remobilize) cost in days of resource cost
    "standby_share": 0.5,          # ASSUMPTION standby cost share of the activity resource rate
    "alpha_route": 6.0,            # ASSUMPTION transit days per GSCPI unit at the episode peak
    "K_links": 6,                  # rule: linked activities per project
    "min_need_day": 80.0,          # rule: earliest eligible need date (working days)
    "est_sigma": 0.3,              # ASSUMPTION error of disruption-duration forecasts (log scale)
    "link_rule": "keyword",        # rule: procured-item keywords, cost-ranked; 'cost' = top fixed cost only
    "max_float": 10.0,             # rule: linked activities are near-critical (baseline total float, working days)
    "ls_quantile": 0.5,            # design: planners protect the latest start at this quantile of sampled float
    "ls_samples": 300,             # design: Monte Carlo samples of activity durations for the protected latest start
    "prod_hedge": 0.2,             # design: production-rate elasticity in the hedge scenarios (rate x f^-0.2)
}


def gscpi_inputs():
    g = json.loads((ROOT / "results/gscpi.json").read_text())
    return float(g["weekly_persistence_from_calibration"]), [float(x) for x in g["peaks_all"]]


@dataclass
class Instance:
    code: str
    name: str
    K: int
    P: np.ndarray
    rel: np.ndarray
    rS: np.ndarray
    LSs: np.ndarray
    rLS: np.ndarray
    sbLS: np.ndarray
    rmLS: np.ndarray
    pcum: np.ndarray
    plan_done: np.ndarray
    link_idx_LS: np.ndarray      # network index of linked activity by LS rank
    dur: np.ndarray
    risk: np.ndarray
    order: np.ndarray
    ptr: np.ndarray
    pidx: np.ndarray
    ptyp: np.ndarray
    plag: np.ndarray
    release: np.ndarray
    bac: float
    pd: float
    wbar: float
    cd: float
    meta: dict = field(default_factory=dict)


def csr_preds(preds):
    code = {"FS": 0, "SS": 1, "FF": 2, "SF": 3}
    ptr = [0]
    pidx, ptyp, plag = [], [], []
    for rels in preds:
        for (i, t, l) in rels:
            pidx.append(i)
            ptyp.append(code[t])
            plag.append(l)
        ptr.append(len(pidx))
    return (np.array(ptr, dtype=np.int64), np.array(pidx, dtype=np.int64), np.array(ptyp, dtype=np.int64),
            np.array(plag, dtype=np.float64))


# Manufactured or procured items that typically arrive as supplier deliveries (Dutch, French, English names)
PROCURED_KEYWORDS = [
    "staal", "steel", "acier", "metal", "gevel", "facade", "façade", "parement", "ramen", "raam", "window", "fenêtre",
    "deur", "door", "schrijnwerk", "menuiserie", "joinery", "dak", "roof", "toiture", "lift", "elevator", "technieken",
    "technique", "klimatisatie", "ventilatie", "ventilat", "hvac", "installeren", "install", "leveren",
    "prefab", "precast", "glas", "glazing", "bekleding", "cladding", "houten kit", "houtskelet", "timber", "pomp",
    "pump", "kabel", "cable", "elektr", "sanitair", "sanitaire", "verwarming", "uitkraging", "rookkoepel", "borden",
    "plafond", "uitrusting", "equipment", "ruimerbrug", "schuiven", "buispalen", "scheepvaartseinen", "turbine",
    "wec", "generator", "transform", "branddetectie", "meubilair",
]


def is_procured(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in PROCURED_KEYWORDS)


def select_links(pr, cfg, explicit_ids=None, tf=None):
    es = np.nan_to_num(pr.es_sheet, nan=0.0)
    if explicit_ids is not None:
        idx = [int(np.where(pr.ids == a)[0][0]) for a in explicit_ids]
        return np.array(sorted(idx, key=lambda k: es[k]), dtype=int)
    elig = [k for k in range(len(pr.dur)) if pr.dur[k] > 0 and es[k] >= cfg["min_need_day"] and pr.fixed_cost[k] > 0
            and (tf is None or tf[k] <= cfg["max_float"] + 1e-9)]
    K = int(cfg["K_links"])
    if cfg.get("link_rule", "keyword") == "keyword":
        kw = sorted([k for k in elig if is_procured(pr.names[k])], key=lambda k: -pr.fixed_cost[k])[:K]
        if len(kw) < 4:
            rest = sorted([k for k in elig if k not in kw], key=lambda k: -pr.fixed_cost[k])
            kw = kw + rest[: max(0, 4 - len(kw))]
        sel = kw
    else:
        sel = sorted(elig, key=lambda k: -pr.fixed_cost[k])[:K]
    return np.array(sorted(sel, key=lambda k: es[k]), dtype=int)


def build_instance(pr, cfg=None, explicit_ids=None) -> Instance | None:
    cfg = {**DEFAULTS, **(cfg or {})}
    if cfg["phi_w"] is None:
        cfg["phi_w"], _ = gscpi_inputs()
    release = np.nan_to_num(pr.es_sheet, nan=0.0)
    es, ef = cpm_forward(pr.dur, pr.preds, pr.order, release=release)
    pdur = float(ef.max())
    ls, lf = cpm_backward(pr.dur, pr.preds, pr.order, pdur)
    links = select_links(pr, cfg, explicit_ids, tf=ls - es)
    K = len(links)
    if K < 4:
        return None
    # protected latest start: quantile of the float of each linked activity over sampled activity durations
    rng = np.random.default_rng(7)
    lo, ml, hi = pr.risk[:, 0] / 100.0, pr.risk[:, 1] / 100.0, pr.risk[:, 2] / 100.0
    lo2, hi2 = np.minimum(lo, ml), np.maximum(hi, ml)
    span = np.maximum(hi2 - lo2, 1e-9)
    fcm = np.clip((ml - lo2) / span, 0.0, 1.0)
    slack = np.zeros((int(cfg["ls_samples"]), len(links)))
    for q in range(slack.shape[0]):
        uu = rng.random(len(lo))
        dm = np.where(uu < fcm, lo2 + np.sqrt(uu * span * (ml - lo2)), hi2 - np.sqrt((1 - uu) * span * (hi2 - ml)))
        dm = np.where(hi2 - lo2 < 1e-9, ml, dm)
        d = pr.dur * dm
        es_q, ef_q = cpm_forward(d, pr.preds, pr.order, release=release)
        ls_q, _ = cpm_backward(d, pr.preds, pr.order, float(ef_q.max()))
        slack[q] = ls_q[links] - es[links]
    ls_prot = es[links] + np.maximum(np.quantile(slack, cfg["ls_quantile"], axis=0), 0.0)
    r = es[links]
    wvals = pr.fixed_cost[links]
    wbar = float(np.mean(wvals))
    rate = pr.cost_per_hour[links] * 8.0
    sb = cfg["standby_share"] * rate
    rm = cfg["remob_days"] * rate
    # rank orders
    order_r = np.argsort(r, kind="stable")
    rS = r[order_r]
    order_ls = np.argsort(ls_prot, kind="stable")
    LSs = ls_prot[order_ls]
    rLS = r[order_ls]
    sbLS = sb[order_ls]
    rmLS = rm[order_ls]
    link_idx_LS = links[order_ls]
    # production plan: backward from need dates
    kappa = 1.0 / cfg["prod_days_per_batch"]
    pipeline = cfg["inland"] + cfg["vessel"] / 2.0 + cfg["T0"] + cfg["C0"] + 1.0 + cfg["A0"]
    p = rS - cfg["buf"] - pipeline
    for k in range(K - 2, -1, -1):
        p[k] = min(p[k], p[k + 1] - 1.0 / kappa)
    rel = p - 1.0 / kappa
    if rel[0] < 0:
        return None
    horizon = int(max(rS.max(), pdur) + 1500)
    pcum = np.zeros(horizon)
    for t in range(horizon):
        pcum[t] = float(np.sum(p <= t))
    P = np.zeros(NP)
    P[cr.P_KAPPA] = kappa
    P[cr.P_KAPPA_ALT] = 1.0 / cfg["alt_prod_days_per_batch"]
    P[cr.P_INLAND] = cfg["inland"]
    P[cr.P_VESSEL] = cfg["vessel"]
    P[cr.P_T0] = cfg["T0"]
    P[cr.P_C0] = cfg["C0"]
    P[cr.P_A0] = cfg["A0"]
    P[cr.P_AREV] = cfg["a_rev"]
    P[cr.P_DR] = cfg["dR"]
    P[cr.P_THETA] = cfg["theta"]
    P[cr.P_FE] = cfg["fe"]
    P[cr.P_COST_EXP] = cfg["cost_exp"]
    P[cr.P_COST_RER] = cfg["cost_rer"]
    P[cr.P_PI_ALT] = cfg["pi_alt"]
    P[cr.P_FC] = cfg["f_c"]
    P[cr.P_Q0] = cfg["Q0"]
    P[cr.P_MARGIN] = cfg["margin"]
    P[cr.P_PEN_D] = cfg["pen_d"]
    P[cr.P_PEN_CAP] = cfg["pen_cap"]
    P[cr.P_EBAR] = cfg["e_bar"] * wbar
    P[cr.P_RHO0] = cfg["rho0"]
    P[cr.P_RHO1] = cfg["rho1"]
    P[cr.P_HU] = cfg["h_u"]
    P[cr.P_PHIW] = cfg["phi_w"]
    P[cr.P_WBAR] = wbar
    bac = float(pr.bac) if np.isfinite(pr.bac) else float(np.sum(pr.total_cost))
    P[cr.P_CD] = cfg["cd_share"] * bac
    P[cr.P_BUF] = cfg["buf"]
    P[cr.P_UPSHARED] = cfg["upshared"]
    P[cr.P_MDAYS] = cfg["m_days"]
    P[cr.P_REMOB] = cfg["remob_days"]
    P[cr.P_PRODHEDGE] = cfg["prod_hedge"]
    ptr, pidx, ptyp, plag = csr_preds(pr.preds)
    return Instance(code=pr.code, name=pr.name, K=K, P=P, rel=rel, rS=rS, LSs=LSs, rLS=rLS, sbLS=sbLS, rmLS=rmLS,
                    pcum=pcum, plan_done=p, link_idx_LS=link_idx_LS, dur=pr.dur.copy(), risk=pr.risk.copy(),
                    order=pr.order.copy(), ptr=ptr, pidx=pidx, ptyp=ptyp, plag=plag, release=release, bac=bac,
                    pd=pdur, wbar=wbar, cd=float(P[cr.P_CD]),
                    meta={"links": [int(pr.ids[k]) for k in links], "link_names": [pr.names[k] for k in links],
                          "total_float": [float(x) for x in (ls[links] - es[links])],
                          "protected_float": [float(x) for x in (ls_prot - es[links])], "cfg": cfg})


# ------------------------------------------------------------------------------------------
# Scenarios (common random numbers per instance and replication)
# ------------------------------------------------------------------------------------------
@dataclass
class Scenario:
    shocks: np.ndarray        # (n,5): type, start, dur_true, severity/peak, dur_est
    noise_tr: np.ndarray
    noise_cl: np.ndarray
    noise_in: np.ndarray
    noise_prod: np.ndarray
    alt_q: float
    dur_mult: np.ndarray      # activity duration multipliers
    seed: int


def make_scenario(inst: Instance, seed: int, cfg=None, n_types=None, spec=None) -> Scenario:
    """Random compound disruption (2 or 3 shocks of distinct types) unless spec gives explicit shocks."""
    cfg = {**DEFAULTS, **(cfg or {})}
    rng = np.random.default_rng(seed)
    _, peaks = gscpi_inputs()
    K = inst.K
    H = len(inst.pcum)
    noise_tr = np.exp(rng.normal(0.0, 0.15, K + 4))
    noise_cl = np.exp(rng.normal(0.0, 0.3, K + 4))
    noise_in = np.exp(rng.normal(0.0, 0.3, K + 4))
    noise_prod = np.exp(rng.normal(0.0, 0.2, H))
    alt_q = float(inst.P[cr.P_Q0] * np.exp(rng.normal(0.0, 0.25)))
    # activity duration multipliers from the DSLIB risk profile (triangular, percent of baseline)
    lo, ml, hi = inst.risk[:, 0] / 100.0, inst.risk[:, 1] / 100.0, inst.risk[:, 2] / 100.0
    lo2 = np.minimum(lo, ml)
    hi2 = np.maximum(hi, ml)
    u = rng.random(len(lo))
    span = np.maximum(hi2 - lo2, 1e-9)
    fc = np.clip((ml - lo2) / span, 0.0, 1.0)
    dm = np.where(u < fc, lo2 + np.sqrt(u * span * (ml - lo2)), hi2 - np.sqrt((1 - u) * span * (hi2 - ml)))
    dm = np.where(hi2 - lo2 < 1e-9, ml, dm)
    if spec is None:
        n = int(rng.integers(2, 4)) if n_types is None else int(n_types)
        types = rng.permutation(3)[:n]
        t1 = float(np.round(rng.uniform(inst.rel[0], inst.plan_done[-1])))
        sh = []
        t = t1
        for i, ty in enumerate(types):
            if i > 0:
                t = float(np.round(t + rng.uniform(5, 50)))
            if ty == 0:
                sev, dur = rng.uniform(0.5, 0.95), rng.uniform(10, 30)
            elif ty == 1:
                sev, dur = cfg["alpha_route"] * peaks[int(rng.integers(len(peaks)))], rng.uniform(10, 40)
            else:
                sev, dur = rng.uniform(0.6, 0.9), rng.uniform(10, 30)
            dur = float(np.round(dur))
            est = float(max(5.0, np.round(dur * np.exp(rng.normal(0.0, cfg["est_sigma"])))))
            sh.append([float(ty), t, dur, float(sev), est])
        shocks = np.array(sh, dtype=np.float64)
    else:
        shocks = np.array(spec, dtype=np.float64).reshape(-1, 5)
    return Scenario(shocks=shocks, noise_tr=noise_tr, noise_cl=noise_cl, noise_in=noise_in, noise_prod=noise_prod,
                    alt_q=alt_q, dur_mult=dm, seed=seed)


# ------------------------------------------------------------------------------------------
# Network evaluation
# ------------------------------------------------------------------------------------------
@njit(cache=True)
def cpm_release_csr(dur, order, ptr, pidx, ptyp, plag, release, supply_idx, supply_t):
    n = dur.shape[0]
    es = np.zeros(n)
    ef = np.zeros(n)
    wait = np.zeros(supply_idx.shape[0])
    for jj in range(n):
        j = order[jj]
        s = release[j]
        for q in range(ptr[j], ptr[j + 1]):
            i = pidx[q]
            t = ptyp[q]
            lag = plag[q]
            if t == 0:
                v = ef[i] + lag
            elif t == 1:
                v = es[i] + lag
            elif t == 2:
                v = ef[i] + lag - dur[j]
            else:
                v = es[i] + lag - dur[j]
            if v > s:
                s = v
        for m in range(supply_idx.shape[0]):
            if supply_idx[m] == j:
                if supply_t[m] > s:
                    wait[m] = supply_t[m] - s
                    s = supply_t[m]
        es[j] = s
        ef[j] = s + dur[j]
    fin = 0.0
    for j in range(n):
        if ef[j] > fin:
            fin = ef[j]
    return fin, wait


# ------------------------------------------------------------------------------------------
# Policies
# ------------------------------------------------------------------------------------------
GRID = {
    "E": np.array([0.0, 0.5, 1.0, 2.0, 4.0]),
    "U": np.array([0.0, 0.25, 0.5]),
    "PHI": np.array([0.0, 0.25, 0.5, 0.75, 0.9, 1.0]),
    "B": np.array([0.0, 0.02, 0.05, 0.10, 0.20]),
}
T_CAP = 700.0
# Three-point Gauss-Hermite quadrature on the log-scale forecast error of disruption durations (sigma = est_sigma)
_GH = np.sqrt(3.0)
HEDGE = {"scales": np.exp(DEFAULTS["est_sigma"] * _GH * np.array([-1.0, 0.0, 1.0])),
         "weights": np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])}
POINT = {"scales": np.array([1.0]), "weights": np.array([1.0])}


def hedge(p):
    return HEDGE if p.get("hedge", True) else POINT
POLICIES = ["naive", "practice", "static_share", "static_bonus", "static_two_part", "nonstrategic_mpc",
            "trigger_rules", "rssp_memoryless", "rssp", "centralized"]


def _opts(S, inst, allow_alt=True):
    exp_opts = np.array([0.0, 1.0])
    route_issue = S[cr.RDELTA] > 0.0 or S[cr.R_END] > S[cr.T]
    rer_opts = np.array([0.0, 1.0]) if route_issue else np.array([0.0])
    remaining_primary = int(S[cr.KP] - S[cr.NPROD_P])
    if allow_alt and S[cr.ALT_N] <= 0 and remaining_primary >= 1:
        alt_opts = np.array([float(x) for x in range(0, min(2, remaining_primary) + 1)])
    else:
        alt_opts = np.array([0.0])
    return exp_opts, rer_opts, alt_opts


def predicted_lateness(S, Bt, inst, dsh, dno, memoryless=0):
    """Planner's view of approval times with no supplier effort and no contractor action."""
    Sx = S.copy()
    Bx = Bt.copy()
    t0 = Sx[cr.T]
    Sx[cr.OUT_END] = max(Sx[cr.OUT_END_EST], t0 + 5.0) if Sx[cr.OUT_END] > t0 else Sx[cr.OUT_END]
    Sx[cr.R_END] = max(Sx[cr.R_END_EST], t0 + 5.0) if Sx[cr.R_END] > t0 else Sx[cr.R_END]
    Sx[cr.A_END] = max(Sx[cr.A_END_EST], t0 + 5.0) if Sx[cr.A_END] > t0 else Sx[cr.A_END]
    advance(Sx, Bx, inst.P, inst.rel, inst.K, t0 + T_CAP, dsh, 0, dno, dno, dno, dno, memoryless, 1, 0.0, 0.0, 0.0,
            0.0, t0, inst.pcum)
    appr, src = approvals_sorted(Bx, inst.K)
    appr = np.where(appr >= cr.INF, t0 + T_CAP + 60.0, appr)
    return float(np.max(appr - inst.rS)), appr


class Policy:
    """Contract and operational decisions of the contractor at each weekly epoch.

    decide() returns (phi, bvec, mode, exp, rer, nalt, imposed_e, imposed_u); imposed_* = -1 means the supplier
    best-responds to the contract in force.
    """

    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}
        self.reset(1)

    def reset(self, K):
        self.tau = None  # duration of the contractor's expediting and rerouting announced with the last decision
        self.committed = np.zeros(K)
        self.bonus_min = 0.0
        self.mode_fixed = -1

    def decide(self, S, Bt, inst, dsh, dno):
        nm = self.name
        p = self.params
        K = inst.K
        zero = np.zeros(K)
        if nm == "naive":
            return 0.0, zero, 0, 0.0, 0.0, 0, -1.0, -1.0
        if nm in ("practice", "static_share", "static_bonus", "static_two_part"):
            lat, _ = predicted_lateness(S, Bt, inst, dsh, dno)
            ex = 1.0 if lat > 0 else 0.0
            phi = p.get("phi", 0.0) if nm in ("static_share", "static_two_part") else 0.0
            bonus = p.get("bonus", 0.0) if nm in ("static_bonus", "static_two_part") else 0.0
            mode = int(p.get("mode", 0))
            return phi, np.full(K, bonus), mode, ex, 0.0, 0, -1.0, -1.0
        if nm == "trigger_rules":
            lat, _ = predicted_lateness(S, Bt, inst, dsh, dno)
            ex = 1.0 if lat > p.get("th_exp", 0.0) else 0.0
            rr = 1.0 if S[cr.RDELTA] > p.get("th_rer", 6.0) else 0.0
            backlog = inst.pcum[min(int(S[cr.T]), len(inst.pcum) - 1)] - S[cr.Y]
            nalt = 1 if (S[cr.ALT_N] <= 0 and backlog >= p.get("th_alt", 1.0) and S[cr.KP] - S[cr.NPROD_P] >= 1) else 0
            return 0.0, zero, 0, ex, rr, nalt, -1.0, -1.0
        exp_opts, rer_opts, alt_opts = _opts(S, inst)
        if nm in ("nonstrategic_mpc", "centralized"):
            h = hedge(p)
            Eg = GRID["E"] * inst.P[cr.P_EBAR]
            res, best = None, np.inf
            for tau, Pt in _lever_blocks(inst.P, p):
                if USE_FAST:
                    phys, appr, srcs = build_cache(S, Bt, Pt, inst.rel, K, inst.rS, inst.LSs, inst.rLS, inst.sbLS,
                                                   inst.rmLS, T_CAP, Eg, GRID["U"], exp_opts, rer_opts, alt_opts, dsh,
                                                   dno, inst.pcum, h["scales"], h["weights"], 0)
                    r_t, v_t = central_fast(phys, appr, srcs, Pt, K, inst.rS, inst.LSs, Eg, GRID["U"], exp_opts,
                                            rer_opts, alt_opts, h["weights"])
                else:
                    r_t, v_t = central_solve(S, Bt, Pt, inst.rel, K, inst.rS, inst.LSs, inst.rLS, inst.sbLS,
                                             inst.rmLS, T_CAP, Eg, GRID["U"], exp_opts, rer_opts, alt_opts,
                                             dsh, dno, inst.pcum, h["scales"], h["weights"])
                if res is None or v_t < best - 1e-6:
                    res, best, self.tau = r_t, v_t, tau
            if nm == "centralized":
                return 0.0, zero, 0, res[2], res[3], int(res[4]), res[0], res[1]
            return 0.0, zero, 0, res[2], res[3], int(res[4]), -1.0, -1.0
        if nm in ("rssp", "rssp_memoryless"):
            mem = 1 if nm == "rssp_memoryless" else 0
            masks, mmode = bonus_masks(S, Bt, inst, dsh, dno, mem, near_term=p.get("near_term", False))
            Pp = inst.P
            if p.get("delay_premium", 0.0):
                Pp = inst.P.copy()
                Pp[cr.P_DPREM] = float(p["delay_premium"])
                Pp[cr.P_DTHR] = float(p.get("delay_threshold", 0.0))
            if p.get("penalty_discount", 0.0):
                Pp = Pp.copy()
                Pp[cr.P_PDISC] = float(p["penalty_discount"])
            Eg = GRID["E"] * inst.P[cr.P_EBAR]
            h = hedge(p)
            res, best = None, np.inf
            for tau, Pt in _lever_blocks(Pp, p):
                if USE_FAST:
                    phys, appr, srcs = build_cache(S, Bt, Pt, inst.rel, K, inst.rS, inst.LSs, inst.rLS, inst.sbLS,
                                                   inst.rmLS, T_CAP, Eg, GRID["U"], exp_opts, rer_opts, alt_opts, dsh,
                                                   dno, inst.pcum, h["scales"], h["weights"], mem)
                    r_t, v_t = leader_fast(phys, appr, srcs, Pt, K, inst.rS, inst.LSs, Eg, GRID["U"], GRID["PHI"],
                                           GRID["B"], masks, mmode, self.committed, self.bonus_min, self.mode_fixed,
                                           exp_opts, rer_opts, alt_opts, h["weights"])
                else:
                    r_t, v_t = leader_solve(S, Bt, Pt, inst.rel, K, inst.rS, inst.LSs, inst.rLS, inst.sbLS, inst.rmLS,
                                            T_CAP, Eg, GRID["U"], GRID["PHI"], GRID["B"], masks, mmode,
                                            self.committed, self.bonus_min, self.mode_fixed, exp_opts, rer_opts,
                                            alt_opts, dsh, dno, inst.pcum, mem, h["scales"], h["weights"])
                if res is None or v_t < best - 1e-6:
                    res, best, self.tau = r_t, v_t, tau
            phi, bonus, mode, mi = float(res[0]), float(res[1]), int(res[2]), int(res[8])
            if bonus > 0 and mi >= 0:
                self.committed = np.maximum(self.committed, bonus * masks[mi])
                self.bonus_min = max(self.bonus_min, bonus)
                if self.mode_fixed < 0:
                    self.mode_fixed = mode
            mode = self.mode_fixed if self.mode_fixed >= 0 else 0
            return phi, self.committed.copy(), mode, float(res[3]), float(res[4]), int(res[5]), -1.0, -1.0
        raise ValueError(nm)


def _lever_blocks(P, p):
    """Durations (working days) for which the plan keeps expediting and rerouting in force, with the matching
    parameter vectors. Without "taus" the plan uses the fixed commitment block of the parameter register."""
    taus = p.get("taus")
    if not taus:
        return [(float(P[cr.P_MDAYS]), P)]
    out = []
    for tau in taus:
        Pt = P.copy()
        Pt[cr.P_MDAYS] = float(tau)
        out.append((float(tau), Pt))
    return out


def bonus_masks(S, Bt, inst, dsh, dno, memoryless, near_term=False):
    """Offers the contractor can make: all batches still to be produced, or only those forecast to miss the milestone.

    near_term=True limits every offer to batches whose production is released within the commitment block, so
    promises for later batches wait for the information the next epochs bring.
    """
    K = inst.K
    nb = int(S[cr.NB])
    _, appr = predicted_lateness(S, Bt, inst, dsh, dno, memoryless)
    t0 = S[cr.T]
    horizon = t0 + inst.P[cr.P_MDAYS]
    kp = int(S[cr.KP])
    def soon(j):
        if not near_term:
            return True
        # rank j >= nb is the (j - nb)-th batch still to be produced; primary releases are in production order
        k = int(S[cr.NPROD_P]) + (j - nb)
        return k >= kp or inst.rel[min(k, len(inst.rel) - 1)] <= horizon
    future = np.array([1.0 if (j >= nb and soon(j)) else 0.0 for j in range(K)])
    masks, modes = [], []
    for mode, ref in ((0, inst.rS), (1, inst.LSs)):
        risk = np.array([1.0 if (j >= nb and soon(j) and appr[j] > ref[j]) else 0.0 for j in range(K)])
        masks.append(future)
        modes.append(mode)
        if risk.sum() > 0 and not np.array_equal(risk, future):
            masks.append(risk)
            modes.append(mode)
    if future.sum() == 0:
        masks = [np.zeros(K)]
        modes = [0]
    return np.array(masks, dtype=np.float64), np.array(modes, dtype=np.int64)


# ------------------------------------------------------------------------------------------
# Simulation driver
# ------------------------------------------------------------------------------------------
def init_state(inst: Instance):
    S = np.zeros(NS)
    S[cr.C] = 1.0
    S[cr.KP] = inst.K
    S[cr.REC_DAY] = -1.0
    S[cr.FIRST_SHOCK] = -1.0
    S[cr.ALT_DAY] = -1.0
    S[cr.OUT_END] = -1.0
    S[cr.OUT_END_EST] = -1.0
    S[cr.R_END] = -1.0
    S[cr.R_END_EST] = -1.0
    S[cr.A_END] = -1.0
    S[cr.A_END_EST] = -1.0
    Bt = np.zeros((inst.K, NBF))
    Bt[:, cr.B_TAPPR] = cr.INF
    Bt[:, cr.B_TSITE] = cr.INF
    Bt[:, cr.B_TREV] = cr.INF
    return S, Bt


DSH = np.zeros((1, 5))
DNO = np.ones(1)


def simulate(inst: Instance, sc: Scenario, policy: Policy, intervention_delay_days: float = 0.0, record=False):
    K = inst.K
    policy.reset(K)
    S, Bt = init_state(inst)
    hist = []  # (t, phi, bvec, mode)
    t = 0.0
    nsh = sc.shocks.shape[0]
    alt_q = sc.alt_q
    trace = []
    E = GRID["E"] * inst.P[cr.P_EBAR]
    zero = np.zeros(K)
    while S[cr.NAPPR] < K and t < len(inst.pcum) - 10:
        active = S[cr.FIRST_SHOCK] >= 0 and t >= S[cr.FIRST_SHOCK] + intervention_delay_days
        ex = rr = 0.0
        if active:
            phi, bvec, mode, ex, rr, nalt, ie, iu = policy.decide(S, Bt, inst, DSH, DNO)
            if nalt > 0 and S[cr.ALT_N] <= 0:
                S[cr.ALT_N] = nalt
                S[cr.KP] -= nalt
                S[cr.ALT_QEND] = t + alt_q
                S[cr.ALT_QEND_EST] = t + inst.P[cr.P_Q0]
                S[cr.ALT_DAY] = t
            S[cr.PHI] = phi
            Ps = inst.P
            if policy.tau is not None and policy.tau != inst.P[cr.P_MDAYS]:
                Ps = inst.P.copy()
                Ps[cr.P_MDAYS] = policy.tau  # the supplier plans with the lever duration the contractor announces
            if ie >= 0:
                e, u = ie, iu
            else:
                e, u, _, _, _, _ = follower_br(S, Bt, Ps, inst.rel, K, inst.rS, inst.LSs, inst.rLS, inst.sbLS,
                                               inst.rmLS, ex, rr, 0, phi, bvec, mode, T_CAP, E, GRID["U"], DSH, DNO,
                                               inst.pcum, 0, HEDGE["scales"], HEDGE["weights"])
        else:
            phi, bvec, mode, e, u = 0.0, zero, 0, 0.0, 0.0
            S[cr.PHI] = 0.0
            if S[cr.FIRST_SHOCK] >= 0:
                # before a delayed intervention starts, the supplier best-responds to the existing contract
                e, u, _, _, _, _ = follower_br(S, Bt, inst.P, inst.rel, K, inst.rS, inst.LSs, inst.rLS, inst.sbLS,
                                               inst.rmLS, 0.0, 0.0, 0, 0.0, zero, 0, T_CAP, E, GRID["U"], DSH, DNO,
                                               inst.pcum, 0, HEDGE["scales"], HEDGE["weights"])
        hist.append((t, phi, np.array(bvec, dtype=float).copy(), mode))
        if record:
            trace.append({"t": t, "c": float(S[cr.C]), "Y": float(S[cr.Y]), "delta": float(S[cr.RDELTA]),
                          "a_sev": float(S[cr.A_SEV]), "phi": phi, "bonus_max": float(np.max(bvec)),
                          "bonus_ranks": int(np.sum(np.asarray(bvec) > 0)), "mode": mode, "exp": ex, "rer": rr,
                          "e": e, "u": u, "alt_n": float(S[cr.ALT_N]), "n_appr": float(S[cr.NAPPR]),
                          "n_prod": float(S[cr.NB]), "planned": float(inst.pcum[min(int(t), len(inst.pcum) - 1)])})
        advance(S, Bt, inst.P, inst.rel, K, t + 5.0, sc.shocks, nsh, sc.noise_tr, sc.noise_cl, sc.noise_in,
                sc.noise_prod, 0, 0, e, u, ex, rr, t + 5.0, inst.pcum)
        t += 5.0
    return settle(inst, sc, S, Bt, hist, trace)


def recovery_days(appr, due, first_shock):
    """Days from the first shock until approved deliveries are back on the contractual schedule for good.

    Backlog(t) = planned approvals by t (contractual dates) minus actual approvals by t. Recovery is the last
    day on which the backlog was positive; zero when the disruption never pushed deliveries behind schedule.
    """
    if first_shock < 0:
        return np.nan
    events = np.unique(np.concatenate([appr, due]))
    last_pos = first_shock
    for t in events:
        if t < first_shock:
            continue
        backlog = np.sum(due <= t + 1e-9) - np.sum(appr <= t + 1e-9)
        if backlog > 0:
            # backlog stays positive until the next approval event
            nxt = appr[appr > t + 1e-9]
            last_pos = max(last_pos, float(nxt.min()) if len(nxt) else t)
    return float(last_pos - first_shock)


def settle(inst, sc, S, Bt, hist, trace):
    K = inst.K
    P = inst.P
    appr, src = approvals_sorted(Bt, K)
    wbar = P[cr.P_WBAR]
    hist_t = np.array([h[0] for h in hist]) if hist else np.array([0.0])
    bonus_paid = 0.0
    penalty = 0.0
    for j in range(K):
        if src[j] < 0.5:
            k = int(np.searchsorted(hist_t, appr[j], side="right") - 1)
            k = max(k, 0)
            _, _, bvec, mode = hist[k] if hist else (0, 0, np.zeros(K), 0)
            dj = inst.rS[j] if mode == 0 else inst.LSs[j]
            if bvec[j] > 0 and appr[j] <= dj + 1e-9:
                bonus_paid += bvec[j] * wbar
            late = appr[j] - (inst.rS[j] - P[cr.P_BUF])
            if late > 0:
                penalty += min(P[cr.P_PEN_D] * late, P[cr.P_PEN_CAP]) * wbar
    # network: EDD assignment by latest start; sampled durations shared by every policy
    dur = inst.dur * sc.dur_mult
    supply_idx = inst.link_idx_LS.astype(np.int64)
    fin_s, wait = cpm_release_csr(dur, inst.order, inst.ptr, inst.pidx, inst.ptyp, inst.plag, inst.release,
                                  supply_idx, appr.copy())
    fin_0, _ = cpm_release_csr(dur, inst.order, inst.ptr, inst.pidx, inst.ptyp, inst.plag, inst.release,
                               supply_idx, np.zeros(K))
    delay = max(fin_s - fin_0, 0.0)
    wait_cost = float(np.sum(np.minimum(inst.sbLS * wait, inst.rmLS)))
    n_alt = S[cr.ALT_N]
    effort, surge, expc, rerc, share = S[cr.CE], S[cr.CS], S[cr.CX], S[cr.CR], S[cr.SHARE]
    alt_c = n_alt * (P[cr.P_PI_ALT] + P[cr.P_FC]) * wbar
    contractor = P[cr.P_CD] * delay + wait_cost + bonus_paid + share + expc + rerc + alt_c - penalty
    supplier = bonus_paid + share - effort - surge - penalty - n_alt * (P[cr.P_MARGIN] - P[cr.P_FC]) * wbar
    system = contractor - supplier
    rec = recovery_days(appr, inst.rS - P[cr.P_BUF], S[cr.FIRST_SHOCK])
    out = {
        "contractor_cost": float(contractor), "supplier_profit": float(supplier), "system_cost": float(system),
        "project_delay": float(delay), "wait_cost": wait_cost, "delay_cost": float(P[cr.P_CD] * delay),
        "bonus_paid": float(bonus_paid), "share_paid": float(share), "penalty": float(penalty),
        "effort": float(effort), "surge": float(surge), "expedite": float(expc), "reroute": float(rerc),
        "alt_batches": float(n_alt), "alt_premium": float(n_alt * P[cr.P_PI_ALT] * wbar),
        "intervention_spend": float(effort + surge + expc + rerc + n_alt * P[cr.P_PI_ALT] * wbar),
        "transfers": float(bonus_paid + share + penalty + n_alt * P[cr.P_FC] * wbar),
        "recovery_days": float(rec), "milestones_on_time": float(np.mean(wait <= 1e-9)),
        "last_approval": float(np.max(appr)), "first_shock": float(S[cr.FIRST_SHOCK]),
        "bac": inst.bac, "cd": inst.cd, "wbar": wbar,
    }
    if trace:
        out["trace"] = trace
        out["approvals"] = appr.tolist()
    return out
