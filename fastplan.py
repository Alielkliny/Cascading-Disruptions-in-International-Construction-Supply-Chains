"""Exact fast path for the planning searches.

The physical trajectory of the planning model (production, sailings, approvals, recovery and logistics spending)
depends only on the supplier's effort and surge, the contractor's logistics levers, the reallocation order and the
duration scenario. The cost share, bonus vector and milestone enter only the payments. Each trajectory is therefore
simulated once per weekly review and every contract option is priced from the cache, with the same arithmetic in
the same order as core.plan_payoffs and core.plan_payoffs_h, so decisions are identical to the direct search.
"""
import numpy as np
from numba import njit

from core import (INF, T, OUT_END, OUT_END_EST, R_END, R_END_EST, A_END, A_END_EST, ALT_N, ALT_QEND, ALT_QEND_EST,
                  KP, PHI, CE, CS, CX, CR, P_Q0, P_MDAYS, P_KAPPA, P_KAPPA_ALT, P_PRODHEDGE, P_WBAR, P_BUF, P_PEN_D,
                  P_PEN_CAP, P_PI_ALT, P_FC, P_MARGIN, P_CD, P_DPREM, P_DTHR, P_PDISC, advance,
                  approvals_sorted)

NPHYS = 8  # effort, surge, expc, rerc, est_delay, wait_cost, penalty, t0


@njit(cache=True)
def _trajectory(S0, B0, P, rel, K, rS, LSs, rLS, sbLS, rmLS, e, u, ex, rr, n_alt_new, memoryless, t_cap, dsh, dno,
                pcum, phys, appr_out, src_out):
    S = S0.copy()
    Bt = B0.copy()
    t0 = S[T]
    S[OUT_END] = max(S[OUT_END_EST], t0 + 5.0) if S[OUT_END] > t0 else S[OUT_END]
    S[R_END] = max(S[R_END_EST], t0 + 5.0) if S[R_END] > t0 else S[R_END]
    S[A_END] = max(S[A_END_EST], t0 + 5.0) if S[A_END] > t0 else S[A_END]
    if S[ALT_N] > 0 and S[ALT_QEND] > t0:
        S[ALT_QEND] = max(S[ALT_QEND_EST], t0 + 5.0)
    if n_alt_new > 0:
        S[ALT_N] += n_alt_new
        S[KP] -= n_alt_new
        S[ALT_QEND] = t0 + P[P_Q0]
    S[PHI] = 0.0  # the cost share only feeds an accumulator that the payoffs do not read
    ce0, cs0, cx0, cr0 = S[CE], S[CS], S[CX], S[CR]
    advance(S, Bt, P, rel, K, t0 + t_cap, dsh, 0, dno, dno, dno, dno, memoryless, 1, e, u, ex, rr, t0 + P[P_MDAYS], pcum)
    appr, src = approvals_sorted(Bt, K)
    for j in range(K):
        if appr[j] >= INF:
            appr[j] = t0 + t_cap + 60.0
    effort = S[CE] - ce0
    surge = S[CS] - cs0
    expc = S[CX] - cx0
    rerc = S[CR] - cr0
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
    wbar = P[P_WBAR]
    penalty = 0.0
    for j in range(K):
        if appr[j] < t0 - 1e-9:
            continue
        if src[j] < 0.5:
            late = appr[j] - (rS[j] - P[P_BUF])
            if late > 0:
                pen = P[P_PEN_D] * late
                if pen > P[P_PEN_CAP]:
                    pen = P[P_PEN_CAP]
                penalty += pen * wbar
    phys[0] = effort
    phys[1] = surge
    phys[2] = expc
    phys[3] = rerc
    phys[4] = est_delay
    phys[5] = wait_cost
    phys[6] = penalty
    phys[7] = t0
    for j in range(K):
        appr_out[j] = appr[j]
        src_out[j] = src[j]


@njit(cache=True)
def build_cache(S0, B0, P, rel, K, rS, LSs, rLS, sbLS, rmLS, t_cap, Egrid, Ugrid, exp_opts, rer_opts, alt_opts,
                dsh, dno, pcum, scales, weights, memoryless):
    nx, ny, nz = exp_opts.shape[0], rer_opts.shape[0], alt_opts.shape[0]
    ni, nj, nq = Egrid.shape[0], Ugrid.shape[0], scales.shape[0]
    phys = np.zeros((nx, ny, nz, ni, nj, nq, NPHYS))
    appr = np.zeros((nx, ny, nz, ni, nj, nq, K))
    src = np.zeros((nx, ny, nz, ni, nj, nq, K))
    t0 = S0[T]
    for q in range(nq):
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
        Pq[P_KAPPA] = P[P_KAPPA] * (f ** (-P[P_PRODHEDGE]))
        Pq[P_KAPPA_ALT] = P[P_KAPPA_ALT] * (f ** (-P[P_PRODHEDGE]))
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    for i in range(ni):
                        for j in range(nj):
                            _trajectory(S, B0, Pq, rel, K, rS, LSs, rLS, sbLS, rmLS, Egrid[i], Ugrid[j], exp_opts[x],
                                        rer_opts[y], alt_opts[z], memoryless, t_cap, dsh, dno, pcum,
                                        phys[x, y, z, i, j, q], appr[x, y, z, i, j, q], src[x, y, z, i, j, q])
    return phys, appr, src


@njit(cache=True)
def pay_h(phys, appr, src, x, y, z, i, j, P, K, rS, LSs, n_alt_new, phi, bvec, mode, weights):
    """Expected payoffs of one option from the cache (same arithmetic and order as plan_payoffs_h)."""
    c_acc = 0.0
    s_acc = 0.0
    y_acc = 0.0
    d_acc = 0.0
    wbar = P[P_WBAR]
    for q in range(weights.shape[0]):
        ph = phys[x, y, z, i, j, q]
        t0 = ph[7]
        ap = appr[x, y, z, i, j, q]
        sr = src[x, y, z, i, j, q]
        bonus_paid = 0.0
        for k in range(K):
            if ap[k] < t0 - 1e-9:
                continue
            if sr[k] < 0.5:
                dj = rS[k] if mode == 0 else LSs[k]
                if bvec[k] > 0 and ap[k] <= dj + 1e-9:
                    bonus_paid += bvec[k] * wbar
        effort, surge, expc, rerc = ph[0], ph[1], ph[2], ph[3]
        est_delay, wait_cost, penalty = ph[4], ph[5], ph[6]
        alt_cost_c = n_alt_new * (P[P_PI_ALT] + P[P_FC]) * wbar
        alt_loss_s = n_alt_new * (P[P_MARGIN] - P[P_FC]) * wbar
        delay_cost = P[P_CD] * max(est_delay, 0.0) + P[P_CD] * P[P_DPREM] * max(est_delay - P[P_DTHR], 0.0)
        contractor = delay_cost + wait_cost + bonus_paid + phi * (effort + surge) + expc + rerc + alt_cost_c - penalty
        supplier = bonus_paid - penalty - (1.0 - phi) * (effort + surge) - alt_loss_s
        system = contractor - supplier
        contractor += P[P_PDISC] * penalty  # planner-only: penalty income not relied on (zero in settlement)
        w = weights[q]
        c_acc += w * contractor
        s_acc += w * supplier
        y_acc += w * system
        d_acc += w * est_delay
    return c_acc, s_acc, y_acc, d_acc


@njit(cache=True)
def follower_cached(phys, appr, src, x, y, z, P, K, rS, LSs, n_alt_new, phi, bvec, mode, weights, Egrid, Ugrid):
    best_s = -1e30
    best_c = 1e30
    be = 0.0
    bu = 0.0
    out_c = 0.0
    for i in range(Egrid.shape[0]):
        for j in range(Ugrid.shape[0]):
            c, s, sy, d = pay_h(phys, appr, src, x, y, z, i, j, P, K, rS, LSs, n_alt_new, phi, bvec, mode, weights)
            if s > best_s + 1e-6 or (abs(s - best_s) <= 1e-6 and c < best_c):
                best_s = s
                best_c = c
                be = Egrid[i]
                bu = Ugrid[j]
                out_c = c
    return be, bu, out_c


@njit(cache=True)
def leader_fast(phys, appr, src, P, K, rS, LSs, Egrid, Ugrid, PHIgrid, Bgrid, masks, mask_mode, committed, bonus_min,
                mode_fixed, exp_opts, rer_opts, alt_opts, weights):
    best = 1e30
    res = np.zeros(9)
    bvec = np.zeros(K)
    nm = masks.shape[0]
    for a in range(PHIgrid.shape[0]):
        phi = PHIgrid[a]
        for b in range(Bgrid.shape[0]):
            bonus = Bgrid[b]
            if bonus > 0 and bonus < bonus_min - 1e-12:
                continue
            for m in range(nm if bonus > 0 else 1):
                mode = mask_mode[m] if bonus > 0 else (mode_fixed if mode_fixed >= 0 else 0)
                if bonus > 0 and mode_fixed >= 0 and mode != mode_fixed:
                    continue
                for j in range(K):
                    v = bonus * masks[m, j] if bonus > 0 else 0.0
                    bvec[j] = v if v > committed[j] else committed[j]
                for x in range(exp_opts.shape[0]):
                    for y in range(rer_opts.shape[0]):
                        for z in range(alt_opts.shape[0]):
                            be, bu, c = follower_cached(phys, appr, src, x, y, z, P, K, rS, LSs, alt_opts[z], phi,
                                                        bvec, mode, weights, Egrid, Ugrid)
                            if c < best - 1e-6:
                                best = c
                                res[0] = phi
                                res[1] = bonus
                                res[2] = mode
                                res[3] = exp_opts[x]
                                res[4] = rer_opts[y]
                                res[5] = alt_opts[z]
                                res[6] = be
                                res[7] = bu
                                res[8] = m if bonus > 0 else -1
    return res, best


@njit(cache=True)
def central_fast(phys, appr, src, P, K, rS, LSs, Egrid, Ugrid, exp_opts, rer_opts, alt_opts, weights):
    best = 1e30
    res = np.zeros(5)
    bvec = np.zeros(K)
    for i in range(Egrid.shape[0]):
        for j in range(Ugrid.shape[0]):
            for x in range(exp_opts.shape[0]):
                for y in range(rer_opts.shape[0]):
                    for z in range(alt_opts.shape[0]):
                        c, s, sy, d = pay_h(phys, appr, src, x, y, z, i, j, P, K, rS, LSs, alt_opts[z], 0.0, bvec, 0,
                                            weights)
                        if sy < best - 1e-6:
                            best = sy
                            res[0] = Egrid[i]
                            res[1] = Ugrid[j]
                            res[2] = exp_opts[x]
                            res[3] = rer_opts[y]
                            res[4] = alt_opts[z]
    return res, best
