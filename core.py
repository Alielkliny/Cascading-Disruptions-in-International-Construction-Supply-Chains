"""Core engine: supply-pipeline dynamics shared by the discrete-event simulation and the planning model.

Time unit: working day. Decision epochs every 5 working days. All money in the project's currency.
The same advance() routine runs the stochastic simulation (true shock ends, sampled noise) and the
deterministic planning model (estimated shock ends, unit noise, no unseen shocks).
"""
from __future__ import annotations

import numpy as np
from numba import njit

INF = 1e18

# ---- state slots -------------------------------------------------------------------------
T, C, OUT_END, OUT_END_EST, Y, NPROD_P, KP, RDELTA, R_END, R_END_EST = range(10)
A_SEV, A_END, A_END_EST, PROG, ALT_N, ALT_QEND, ALT_QEND_EST, YALT, NPROD_A, NB = range(10, 20)
E, U, EXP, RER, CE, CS, CX, CR, NAPPR, PHI = range(20, 30)
SHARE, REC_DAY, FIRST_SHOCK, ALT_DAY, NSTATE_USED = range(30, 35)
NS = 40

# ---- batch fields --------------------------------------------------------------------------
B_SRC, B_TPROD, B_TDEP, B_ROUTE, B_EXP, B_DEP, B_TSITE, B_TREV, B_TAPPR, B_DELTA = range(10)
NBF = 10

# ---- parameter slots -----------------------------------------------------------------------
(P_KAPPA, P_KAPPA_ALT, P_INLAND, P_VESSEL, P_T0, P_C0, P_A0, P_AREV, P_DR, P_THETA,
 P_FE, P_COST_EXP, P_COST_RER, P_PI_ALT, P_FC, P_Q0, P_MARGIN, P_PEN_D, P_PEN_CAP, P_EBAR,
 P_RHO0, P_RHO1, P_HU, P_PHIW, P_WBAR, P_CD, P_BUF, P_UPSHARED, P_MDAYS, P_REMOB, P_PRODHEDGE, P_DPREM,
 P_DTHR, P_PDISC) = range(34)
NP = 34


@njit(cache=True)
def next_departure(t, vessel):
    k = np.ceil(t / vessel)
    return k * vessel


@njit(cache=True)
def rho(e, P):
    if P[P_EBAR] <= 0:
        return P[P_RHO0]
    return P[P_RHO0] + (P[P_RHO1] - P[P_RHO0]) * (1.0 - np.exp(-e / P[P_EBAR]))


@njit(cache=True)
def released_count(rel, kp, t):
    n = 0
    for k in range(kp):
        if rel[k] <= t + 1e-9:
            n += 1
    return n


@njit(cache=True)
def advance(S, Bt, P, rel, K, t_end, shocks, n_shocks, noise_tr, noise_cl, noise_in, noise_prod,
            memoryless, plan_mode, e_block, u_block, exp_block, rer_block, block_end, planned_cum):
    """Advance the pipeline day by day from S[T] to t_end (exclusive) or until all K batches are approved.

    plan_mode=1: deterministic planning (unit noise, estimated ends already written into the true-end slots).
    e_block/u_block/exp_block/rer_block apply until block_end; afterwards no intervention (planning) or
    the values stay as set by the caller (simulation sets block_end beyond the epoch).
    planned_cum: planned cumulative primary output by day (for the recovery-duration measure), may be empty.
    """
    t = S[T]
    while t < t_end and S[NAPPR] < K:
        ti = int(t)
        # --- shock onsets (simulation only) ---
        for i in range(n_shocks):
            if shocks[i, 1] == t:
                typ = int(shocks[i, 0])
                if S[FIRST_SHOCK] < 0:
                    S[FIRST_SHOCK] = t
                if typ == 0:
                    S[C] = S[C] * (1.0 - shocks[i, 3])
                    S[OUT_END] = max(S[OUT_END], t + shocks[i, 2])
                    S[OUT_END_EST] = max(S[OUT_END_EST], t + shocks[i, 4])
                elif typ == 1:
                    S[RDELTA] = S[RDELTA] + shocks[i, 3]
                    S[R_END] = max(S[R_END], t + shocks[i, 2])
                    S[R_END_EST] = max(S[R_END_EST], t + shocks[i, 4])
                else:
                    S[A_SEV] = 1.0 - (1.0 - S[A_SEV]) * (1.0 - shocks[i, 3])
                    S[A_END] = max(S[A_END], t + shocks[i, 2])
                    S[A_END_EST] = max(S[A_END_EST], t + shocks[i, 4])
        in_block = t < block_end
        e = e_block
        u = u_block
        ex = exp_block if in_block else 0.0
        rr = rer_block if in_block else 0.0
        # --- weekly updates ---
        if ti % 5 == 0:
            if t >= S[OUT_END]:
                if memoryless == 1:
                    S[C] = 1.0
                else:
                    S[C] = S[C] + rho(e, P) * (1.0 - S[C])
                    if S[C] > 0.9999:
                        S[C] = 1.0
            if t >= S[R_END]:
                if memoryless == 1:
                    S[RDELTA] = 0.0
                else:
                    S[RDELTA] = S[RDELTA] * P[P_PHIW]
                    if S[RDELTA] < 0.05:
                        S[RDELTA] = 0.0
            if e > 0 and t >= S[OUT_END] and S[C] < 0.9999:
                S[CE] += e
                S[SHARE] += S[PHI] * e
        if t >= S[A_END] and S[A_SEV] > 0:
            S[A_SEV] = 0.0
            if memoryless == 1:
                S[PROG] = 1e6  # planner assumes the pending queue clears at the end of the window
        # --- primary production ---
        kp = int(S[KP])
        rc = released_count(rel, kp, t)
        if S[Y] < rc - 1e-12:
            nz = 1.0 if plan_mode == 1 else noise_prod[min(ti, noise_prod.shape[0] - 1)]
            base = P[P_KAPPA] * S[C] * nz
            add = base * (1.0 + u)
            newy = min(float(rc), S[Y] + add)
            produced = newy - S[Y]
            if u > 0 and add > 0:
                surge_out = produced * u / (1.0 + u)
                cs = surge_out * P[P_HU] * P[P_WBAR]
                S[CS] += cs
                S[SHARE] += S[PHI] * cs
            S[Y] = newy
        while np.floor(S[Y] + 1e-9) > S[NPROD_P] and S[NB] < K:
            b = int(S[NB])
            Bt[b, B_SRC] = 0.0
            Bt[b, B_TPROD] = t
            Bt[b, B_TDEP] = next_departure(t + P[P_INLAND], P[P_VESSEL])
            Bt[b, B_ROUTE] = rr
            Bt[b, B_EXP] = ex
            Bt[b, B_DEP] = 0.0
            Bt[b, B_TSITE] = INF
            Bt[b, B_TREV] = INF
            Bt[b, B_TAPPR] = INF
            S[NPROD_P] += 1.0
            S[NB] += 1.0
        # --- alternative source production ---
        if S[ALT_N] > 0 and t >= S[ALT_QEND] and S[NPROD_A] < S[ALT_N]:
            nz = 1.0 if plan_mode == 1 else noise_prod[min(ti, noise_prod.shape[0] - 1)]
            cap = S[C] if P[P_UPSHARED] > 0.5 else 1.0
            S[YALT] = min(S[ALT_N], S[YALT] + P[P_KAPPA_ALT] * cap * nz)
            while np.floor(S[YALT] + 1e-9) > S[NPROD_A] and S[NB] < K:
                b = int(S[NB])
                Bt[b, B_SRC] = 1.0
                Bt[b, B_TPROD] = t
                Bt[b, B_TDEP] = next_departure(t + P[P_INLAND], P[P_VESSEL])
                Bt[b, B_ROUTE] = rr
                Bt[b, B_EXP] = ex
                Bt[b, B_DEP] = 0.0
                Bt[b, B_TSITE] = INF
                Bt[b, B_TREV] = INF
                Bt[b, B_TAPPR] = INF
                S[NPROD_A] += 1.0
                S[NB] += 1.0
        # --- departures ---
        nb = int(S[NB])
        for b in range(nb):
            if Bt[b, B_DEP] < 0.5 and Bt[b, B_TDEP] <= t + 1e-9:
                # the expedite and reroute flags in force at departure govern the batch
                Bt[b, B_EXP] = ex if ex > Bt[b, B_EXP] else Bt[b, B_EXP]
                Bt[b, B_ROUTE] = rr if rr > Bt[b, B_ROUTE] else Bt[b, B_ROUTE]
                ntr = 1.0 if plan_mode == 1 else noise_tr[b]
                ncl = 1.0 if plan_mode == 1 else noise_cl[b]
                base_tr = P[P_T0] * (P[P_FE] if Bt[b, B_EXP] > 0.5 else 1.0) * ntr
                if Bt[b, B_ROUTE] > 0.5:
                    dl = P[P_DR] + P[P_THETA] * S[RDELTA]
                    S[CR] += P[P_COST_RER] * P[P_WBAR]
                else:
                    dl = S[RDELTA]
                if Bt[b, B_EXP] > 0.5:
                    S[CX] += P[P_COST_EXP] * P[P_WBAR]
                Bt[b, B_DELTA] = dl
                Bt[b, B_TSITE] = t + base_tr + dl + P[P_C0] * ncl
                Bt[b, B_DEP] = 1.0
        # --- approvals: FIFO review queue with daily capacity, then inspection ---
        a = P[P_AREV] * (1.0 - S[A_SEV])
        S[PROG] += a
        while S[PROG] >= 1.0:
            best = -1
            bt = INF
            for b in range(nb):
                if Bt[b, B_TSITE] <= t + 1e-9 and Bt[b, B_TREV] >= INF:
                    if Bt[b, B_TSITE] < bt:
                        bt = Bt[b, B_TSITE]
                        best = b
            if best < 0:
                break
            nin = 1.0 if plan_mode == 1 else noise_in[best]
            Bt[best, B_TREV] = t
            Bt[best, B_TAPPR] = t + P[P_A0] * nin
            S[NAPPR] += 1.0
            S[PROG] -= 1.0
        if S[PROG] > 1.0:
            S[PROG] = 1.0
        # --- recovery-duration bookkeeping (simulation) ---
        if plan_mode == 0 and S[FIRST_SHOCK] >= 0 and S[REC_DAY] < 0 and t > S[FIRST_SHOCK]:
            backlog_ok = True
            if planned_cum.shape[0] > 0:
                pc = planned_cum[min(ti, planned_cum.shape[0] - 1)]
                backlog_ok = (S[Y] + S[YALT]) >= min(pc, S[KP] + S[ALT_N]) - 1e-6
            last_shock = 0.0
            for i in range(n_shocks):
                if shocks[i, 1] > last_shock:
                    last_shock = shocks[i, 1]
            if (t > last_shock and S[C] >= 0.95 and S[RDELTA] <= 0.5 and S[A_SEV] <= 0.0 and backlog_ok):
                S[REC_DAY] = t
        t += 1.0
        S[T] = t
    return S[NAPPR] >= K


@njit(cache=True)
def approvals_sorted(Bt, K):
    a = np.empty(K)
    src = np.empty(K)
    for b in range(K):
        a[b] = Bt[b, B_TAPPR]
        src[b] = Bt[b, B_SRC]
    idx = np.argsort(a)
    return a[idx], src[idx]


@njit(cache=True)
def plan_payoffs(S0, B0, P, rel, K, rS, LSs, rLS, sbLS, rmLS, e, u, ex, rr, n_alt_new, phi, bvec, mode,
                 memoryless, t_cap, dummy_shocks, dummy_noise, planned_cum):
    """Deterministic planning evaluation from the snapshot (S0,B0) under one decision.

    Returns (contractor_cost, supplier_payoff, system_cost, est_delay).
    Only future flows count: approvals already made are excluded from bonus and penalty terms.
    """
    S = S0.copy()
    Bt = B0.copy()
    t0 = S[T]
    # the planner's world: estimated ends in place of true ends
    S[OUT_END] = max(S[OUT_END_EST], t0 + 5.0) if S[OUT_END] > t0 else S[OUT_END]
    S[R_END] = max(S[R_END_EST], t0 + 5.0) if S[R_END] > t0 else S[R_END]
    S[A_END] = max(S[A_END_EST], t0 + 5.0) if S[A_END] > t0 else S[A_END]
    if S[ALT_N] > 0 and S[ALT_QEND] > t0:
        S[ALT_QEND] = max(S[ALT_QEND_EST], t0 + 5.0)
    n_appr0 = S[NAPPR]
    # alternative-source order placed now: reallocate the last undelivered primary batches
    if n_alt_new > 0:
        S[ALT_N] += n_alt_new
        S[KP] -= n_alt_new
        S[ALT_QEND] = t0 + P[P_Q0]
    S[PHI] = phi
    ce0, cs0, cx0, cr0 = S[CE], S[CS], S[CX], S[CR]
    advance(S, Bt, P, rel, K, t0 + t_cap, dummy_shocks, 0, dummy_noise, dummy_noise, dummy_noise, dummy_noise,
            memoryless, 1, e, u, ex, rr, t0 + P[P_MDAYS], planned_cum)
    appr, src = approvals_sorted(Bt, K)
    for j in range(K):
        if appr[j] >= INF:
            appr[j] = t0 + t_cap + 60.0
    effort = S[CE] - ce0
    surge = S[CS] - cs0
    expc = S[CX] - cx0
    rerc = S[CR] - cr0
    # delay estimate (EDD assignment by latest start) and waiting cost
    est_delay = 0.0
    wait_cost = 0.0
    for j in range(K):
        lat = appr[j] - LSs[j]
        if lat > est_delay:
            est_delay = lat
        w = appr[j] - rLS[j]
        if w > 0 and appr[j] > t0:
            wc = sbLS[j] * w
            if wc > rmLS[j]:
                wc = rmLS[j]
            wait_cost += wc
    # bonus and penalty on future approvals (rank-based schedule)
    wbar = P[P_WBAR]
    bonus_paid = 0.0
    penalty = 0.0
    for j in range(K):
        if appr[j] < t0 - 1e-9:  # settled before this epoch: sunk for every decision
            continue
        if src[j] < 0.5:
            dj = rS[j] if mode == 0 else LSs[j]
            if bvec[j] > 0 and appr[j] <= dj + 1e-9:
                bonus_paid += bvec[j] * wbar
            late = appr[j] - (rS[j] - P[P_BUF])
            if late > 0:
                pen = P[P_PEN_D] * late
                if pen > P[P_PEN_CAP]:
                    pen = P[P_PEN_CAP]
                penalty += pen * wbar
    alt_cost_c = n_alt_new * (P[P_PI_ALT] + P[P_FC]) * wbar
    alt_loss_s = n_alt_new * (P[P_MARGIN] - P[P_FC]) * wbar
    # planner-only delay-risk premium (zero in the simulation's settlement)
    delay_cost = P[P_CD] * max(est_delay, 0.0) + P[P_CD] * P[P_DPREM] * max(est_delay - P[P_DTHR], 0.0)
    contractor = delay_cost + wait_cost + bonus_paid + phi * (effort + surge) + expc + rerc + alt_cost_c - penalty
    supplier = bonus_paid - penalty - (1.0 - phi) * (effort + surge) - alt_loss_s
    system = contractor - supplier
    contractor += P[P_PDISC] * penalty  # planner-only: penalty income not relied on (zero in settlement)
    return contractor, supplier, system, est_delay



@njit(cache=True)
def plan_payoffs_h(S0, B0, P, rel, K, rS, LSs, rLS, sbLS, rmLS, e, u, ex, rr, n_alt_new, phi, bvec, mode,
                   memoryless, t_cap, dummy_shocks, dummy_noise, planned_cum, scales, weights):
    """Expected planning payoffs over duration scenarios: remaining disruption and qualification times scaled."""
    c_acc = 0.0
    s_acc = 0.0
    y_acc = 0.0
    d_acc = 0.0
    t0 = S0[T]
    for q in range(scales.shape[0]):
        S = S0.copy()
        f = scales[q]
        if S[OUT_END] > t0:
            S[OUT_END_EST] = t0 + max(S[OUT_END_EST] - t0, 5.0) * f
        if S[R_END] > t0:
            S[R_END_EST] = t0 + max(S[R_END_EST] - t0, 5.0) * f
        if S[A_END] > t0:
            S[A_END_EST] = t0 + max(S[A_END_EST] - t0, 5.0) * f
        if S[ALT_N] > 0 and S[ALT_QEND] > t0:
            S[ALT_QEND_EST] = t0 + max(S[ALT_QEND_EST] - t0, 5.0) * f
        Pq = P.copy()
        # comonotone scenario: longer disruptions come with slower production and the reverse
        Pq[P_KAPPA] = P[P_KAPPA] * (f ** (-P[P_PRODHEDGE]))
        Pq[P_KAPPA_ALT] = P[P_KAPPA_ALT] * (f ** (-P[P_PRODHEDGE]))
        c, s, y, d = plan_payoffs(S, B0, Pq, rel, K, rS, LSs, rLS, sbLS, rmLS, e, u, ex, rr, n_alt_new, phi, bvec,
                                  mode, memoryless, t_cap, dummy_shocks, dummy_noise, planned_cum)
        w = weights[q]
        c_acc += w * c
        s_acc += w * s
        y_acc += w * y
        d_acc += w * d
    return c_acc, s_acc, y_acc, d_acc
