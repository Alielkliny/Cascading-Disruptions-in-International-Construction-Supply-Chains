"""Verification tests: parser, network, pipeline dynamics, game solvers, boundary and extreme conditions."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core as cr  # noqa: E402
import engine as en  # noqa: E402
from dslib import parse_duration, parse_relations, cpm_forward, cpm_backward  # noqa: E402
from solvers import follower_br  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def inst():
    d = pickle.load(open(ROOT / "data/processed/instances.pkl", "rb"))
    return d["case"]


def test_parse_duration_units():
    assert parse_duration("5d") == 5.0
    assert parse_duration("2w") == 10.0
    assert parse_duration("4h") == 0.5
    assert parse_duration("1w 2d") == 7.0
    assert parse_duration("3d 4h") == 3.5
    assert parse_duration(0) == 0.0


def test_parse_relations_types_and_lags():
    r = parse_relations("58FS;53SS+2d;12FF-1w;7SF")
    assert r == [(58, "FS", 0.0), (53, "SS", 2.0), (12, "FF", -5.0), (7, "SF", 0.0)]


def test_cpm_all_relation_types():
    dur = np.array([5.0, 3.0, 4.0, 2.0])
    preds = [[], [(0, "FS", 1.0)], [(0, "SS", 2.0)], [(1, "FF", 0.0), (2, "SF", 3.0)]]
    order = np.array([0, 1, 2, 3])
    es, ef = cpm_forward(dur, preds, order)
    assert es.tolist() == [0.0, 6.0, 2.0, 7.0]  # FF: ef1=9 -> es3=7; SF: es2+3-2=3
    ls, lf = cpm_backward(dur, preds, order, float(ef.max()))
    assert np.all(ls - es >= -1e-9)


def _run(inst, scen_spec, policy="naive", seed=1):
    sc = en.make_scenario(inst, seed, spec=scen_spec)
    return en.simulate(inst, sc, en.Policy(policy))


def test_no_disruption_no_delay_no_cost(inst):
    sc = en.make_scenario(inst, 3, spec=np.zeros((0, 5)))
    sc.noise_tr[:] = 1.0
    sc.noise_cl[:] = 1.0
    sc.noise_in[:] = 1.0
    sc.noise_prod[:] = 1.0
    r = en.simulate(inst, sc, en.Policy("naive"))
    assert r["project_delay"] == pytest.approx(0.0, abs=1e-9)
    assert r["penalty"] == pytest.approx(0.0, abs=1e-6)
    assert r["intervention_spend"] == 0.0
    assert r["recovery_days"] != r["recovery_days"]  # NaN: no shock


def test_deterministic_rerun_identical(inst):
    spec = [[0, float(inst.rel[1]), 20, 0.8, 20], [1, float(inst.rel[2]), 25, 18.0, 25]]
    a = _run(inst, spec, "rssp", seed=11)
    b = _run(inst, spec, "rssp", seed=11)
    assert a["contractor_cost"] == b["contractor_cost"]
    assert a["project_delay"] == b["project_delay"]


def test_delay_monotone_in_supplier_severity(inst):
    base = float(inst.rel[1])
    out = []
    for sev in (0.2, 0.5, 0.8, 0.95):
        out.append(_run(inst, [[0, base, 25, sev, 25]], "naive", seed=5)["project_delay"])
    assert all(out[i] <= out[i + 1] + 1e-9 for i in range(len(out) - 1))


def test_capacity_bounded_and_backlog_nonnegative(inst):
    S, Bt = en.init_state(inst)
    sh = np.array([[0.0, float(inst.rel[0]), 30.0, 0.99, 30.0]])
    ones = np.ones(len(inst.pcum))
    cr.advance(S, Bt, inst.P, inst.rel, inst.K, float(inst.rel[0]) + 60, sh, 1, ones, ones, ones, ones, 0, 0,
               4.0 * inst.P[cr.P_EBAR], 0.5, 0.0, 0.0, 1e9, inst.pcum)
    assert 0.0 <= S[cr.C] <= 1.0
    assert S[cr.Y] >= 0.0
    assert S[cr.Y] <= inst.K + 1e-9


def test_follower_best_response_matches_brute_force(inst):
    S, Bt = en.init_state(inst)
    sh = np.array([[0.0, float(inst.rel[0]) + 2, 20.0, 0.8, 20.0]])
    ones = np.ones(len(inst.pcum))
    cr.advance(S, Bt, inst.P, inst.rel, inst.K, float(inst.rel[0]) + 10, sh, 1, ones, ones, ones, ones, 0, 0,
               0.0, 0.0, 0.0, 0.0, 0.0, inst.pcum)
    E = en.GRID["E"] * inst.P[cr.P_EBAR]
    U = en.GRID["U"]
    bvec = np.full(inst.K, 0.05)
    be, bu, c, s, y, d = follower_br(S, Bt, inst.P, inst.rel, inst.K, inst.rS, inst.LSs, inst.rLS, inst.sbLS,
                                     inst.rmLS, 0.0, 0.0, 0, 0.5, bvec, 0, en.T_CAP, E, U, en.DSH, en.DNO, inst.pcum, 0,
                                     en.HEDGE["scales"], en.HEDGE["weights"])
    best = -1e30
    for e in E:
        for u in U:
            _, sp, _, _ = cr.plan_payoffs_h(S, Bt, inst.P, inst.rel, inst.K, inst.rS, inst.LSs, inst.rLS, inst.sbLS,
                                            inst.rmLS, e, u, 0.0, 0.0, 0, 0.5, bvec, 0, 0, en.T_CAP, en.DSH, en.DNO,
                                            inst.pcum, en.HEDGE["scales"], en.HEDGE["weights"])
            best = max(best, sp)
    assert s == pytest.approx(best, rel=1e-9, abs=1e-6)


def test_transfers_cancel_in_system_cost(inst):
    spec = [[0, float(inst.rel[1]), 25, 0.85, 25]]
    r = _run(inst, spec, "rssp", seed=21)
    resource = r["delay_cost"] + r["wait_cost"] + r["effort"] + r["surge"] + r["expedite"] + r["reroute"]
    alt = r["alt_batches"] * (inst.P[cr.P_PI_ALT] + inst.P[cr.P_MARGIN]) * inst.wbar
    assert r["system_cost"] == pytest.approx(resource + alt, rel=1e-9, abs=1e-6)


def test_supplier_participation_under_offer(inst):
    """Offers only add payments, so the supplier's best-response payoff never falls below the existing contract's."""
    S, Bt = en.init_state(inst)
    sh = np.array([[0.0, float(inst.rel[0]) + 2, 25.0, 0.9, 25.0]])
    ones = np.ones(len(inst.pcum))
    cr.advance(S, Bt, inst.P, inst.rel, inst.K, float(inst.rel[0]) + 10, sh, 1, ones, ones, ones, ones, 0, 0,
               0.0, 0.0, 0.0, 0.0, 0.0, inst.pcum)
    E = en.GRID["E"] * inst.P[cr.P_EBAR]
    args = (S, Bt, inst.P, inst.rel, inst.K, inst.rS, inst.LSs, inst.rLS, inst.sbLS, inst.rmLS, 0.0, 0.0, 0)
    base = follower_br(*args, 0.0, np.zeros(inst.K), 0, en.T_CAP, E, en.GRID["U"], en.DSH, en.DNO, inst.pcum, 0,
                       en.HEDGE["scales"], en.HEDGE["weights"])[3]
    for phi in en.GRID["PHI"]:
        for b in en.GRID["B"]:
            s = follower_br(*args, phi, np.full(inst.K, b), 0, en.T_CAP, E, en.GRID["U"], en.DSH, en.DNO, inst.pcum, 0,
                            en.HEDGE["scales"], en.HEDGE["weights"])[3]
            assert s >= base - 1e-6


def test_memoryless_planner_assumes_full_recovery(inst):
    S, Bt = en.init_state(inst)
    S[cr.C] = 0.4
    S[cr.OUT_END] = 10.0
    S[cr.T] = 10.0
    ones = np.ones(len(inst.pcum))
    S1, B1 = S.copy(), Bt.copy()
    cr.advance(S1, B1, inst.P, inst.rel, inst.K, 16.0, en.DSH, 0, ones, ones, ones, ones, 1, 1, 0.0, 0.0, 0.0, 0.0,
               0.0, inst.pcum)
    S2, B2 = S.copy(), Bt.copy()
    cr.advance(S2, B2, inst.P, inst.rel, inst.K, 16.0, en.DSH, 0, ones, ones, ones, ones, 0, 1, 0.0, 0.0, 0.0, 0.0,
               0.0, inst.pcum)
    assert S1[cr.C] == 1.0
    assert S2[cr.C] < 1.0
