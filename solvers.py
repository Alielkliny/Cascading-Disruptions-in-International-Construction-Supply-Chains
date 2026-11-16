"""Stackelberg solvers on the planning model: supplier best response, contractor leader search, centralized plan."""
from __future__ import annotations

import numpy as np
from numba import njit

from core import plan_payoffs_h


@njit(cache=True)
def follower_br(S0, B0, P, rel, K, rS, LSs, rLS, sbLS, rmLS, ex, rr, nalt, phi, bvec, mode, t_cap,
                Egrid, Ugrid, dsh, dno, pcum, memoryless, scales, weights):
    """Supplier best response (effort, surge) to the contract in force; ties go to the contractor's preference.

    memoryless=0 is the supplier's own (state-aware) view; the ablation's leader predicts it with memoryless=1.
    """
    best_s = -1e30
    best_c = 1e30
    be = 0.0
    bu = 0.0
    out_c = 0.0
    out_sys = 0.0
    out_d = 0.0
    for i in range(Egrid.shape[0]):
        for j in range(Ugrid.shape[0]):
            c, s, sy, d = plan_payoffs_h(S0, B0, P, rel, K, rS, LSs, rLS, sbLS, rmLS, Egrid[i], Ugrid[j], ex, rr,
                                         nalt, phi, bvec, mode, memoryless, t_cap, dsh, dno, pcum, scales, weights)
            if s > best_s + 1e-6 or (abs(s - best_s) <= 1e-6 and c < best_c):
                best_s = s
                best_c = c
                be = Egrid[i]
                bu = Ugrid[j]
                out_c = c
                out_sys = sy
                out_d = d
    return be, bu, out_c, best_s, out_sys, out_d


@njit(cache=True)
def leader_solve(S0, B0, P, rel, K, rS, LSs, rLS, sbLS, rmLS, t_cap, Egrid, Ugrid, PHIgrid, Bgrid, masks, mask_mode,
                 committed, bonus_min, mode_fixed, exp_opts, rer_opts, alt_opts, dsh, dno, pcum, memoryless,
                 scales, weights):
    """Contractor's Stackelberg choice of incentives and operational levers anticipating the supplier's best response.

    masks[m, j] = 1 when the bonus offer m covers approval rank j; mask_mode[m] is its milestone (0 need date,
    1 latest start). committed[j] holds bonus levels already promised, which later offers never reduce.
    """
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
                            be, bu, c, s, sy, d = follower_br(S0, B0, P, rel, K, rS, LSs, rLS, sbLS, rmLS, exp_opts[x],
                                                              rer_opts[y], alt_opts[z], phi, bvec, mode, t_cap, Egrid,
                                                              Ugrid, dsh, dno, pcum, memoryless, scales, weights)
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
def central_solve(S0, B0, P, rel, K, rS, LSs, rLS, sbLS, rmLS, t_cap, Egrid, Ugrid, exp_opts, rer_opts, alt_opts,
                  dsh, dno, pcum, scales, weights):
    """System-cost minimizing plan with direct control of supplier effort (benchmark)."""
    best = 1e30
    res = np.zeros(5)
    bvec = np.zeros(K)
    for i in range(Egrid.shape[0]):
        for j in range(Ugrid.shape[0]):
            for x in range(exp_opts.shape[0]):
                for y in range(rer_opts.shape[0]):
                    for z in range(alt_opts.shape[0]):
                        c, s, sy, d = plan_payoffs_h(S0, B0, P, rel, K, rS, LSs, rLS, sbLS, rmLS, Egrid[i],
                                                     Ugrid[j], exp_opts[x], rer_opts[y], alt_opts[z], 0.0, bvec, 0,
                                                     0, t_cap, dsh, dno, pcum, scales, weights)
                        if sy < best - 1e-6:
                            best = sy
                            res[0] = Egrid[i]
                            res[1] = Ugrid[j]
                            res[2] = exp_opts[x]
                            res[3] = rer_opts[y]
                            res[4] = alt_opts[z]
    return res, best
