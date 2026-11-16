"""Disruption persistence from the NY Fed Global Supply Chain Pressure Index (GSCPI).

Calibration window 1998-01 to 2014-12; validation window 2015-01 to the last month in the file.
Writes probe/premise.json and results/gscpi.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/gscpi/gscpi_data.xlsx"


def load():
    df = pd.read_excel(SRC, "GSCPI Monthly Data", header=None, usecols=[0, 1])
    df.columns = ["date", "gscpi"]
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna().copy()
    df["gscpi"] = df["gscpi"].astype(float)
    return df.reset_index(drop=True)


def ar1(x: np.ndarray):
    y, z = x[1:], x[:-1]
    X = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tcrit = stats.t.ppf(0.975, n - k)
    phi, c = beta[1], beta[0]
    return {
        "phi": float(phi), "phi_se": float(se[1]), "phi_ci95": [float(phi - tcrit * se[1]), float(phi + tcrit * se[1])],
        "const": float(c), "mu": float(c / (1 - phi)), "sigma": float(np.sqrt(s2)), "n": int(n),
        "half_life_months": float(np.log(0.5) / np.log(phi)) if 0 < phi < 1 else None,
    }


def episodes(df: pd.DataFrame, thr: float = 1.0):
    """Contiguous runs above thr: start, peak, months above, months from peak back below thr."""
    out, run = [], []
    for i, v in enumerate(df["gscpi"].to_numpy()):
        if v > thr:
            run.append(i)
        elif run:
            out.append(run)
            run = []
    if run:
        out.append(run)
    eps = []
    g = df["gscpi"].to_numpy()
    for r in out:
        pk = r[int(np.argmax(g[r]))]
        eps.append({"start": str(df["date"][r[0]].date()), "end": str(df["date"][r[-1]].date()),
                    "months_above": len(r), "peak": float(g[pk]), "peak_date": str(df["date"][pk].date()),
                    "months_peak_to_exit": int(r[-1] - pk + 1)})
    return eps


def main():
    df = load()
    sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    cal = df[df["date"] <= "2014-12-31"]
    val = df[df["date"] > "2014-12-31"]
    fit_cal = ar1(cal["gscpi"].to_numpy())
    fit_val = ar1(val["gscpi"].to_numpy())
    fit_all = ar1(df["gscpi"].to_numpy())
    # out-of-sample one-step-ahead coverage in the validation window using calibration parameters
    x = df["gscpi"].to_numpy()
    idx = np.where(df["date"] > "2014-12-31")[0]
    pred = fit_cal["const"] + fit_cal["phi"] * x[idx - 1]
    err = x[idx] - pred
    half = stats.norm.ppf(0.975) * fit_cal["sigma"]
    cover = float(np.mean(np.abs(err) <= half))
    # decay check: after each validation-window peak above 1.0, compare observed multi-month decay
    # with the calibration persistence (phi^h) over 1 to 6 months
    eps_val = episodes(val.reset_index(drop=True))
    decay_rows = []
    for e in eps_val:
        pk = int(np.where(df["date"].dt.date.astype(str) == e["peak_date"])[0][0])
        for h in range(1, 7):
            if pk + h < len(x):
                obs = (x[pk + h] - fit_cal["mu"]) / (x[pk] - fit_cal["mu"])
                decay_rows.append({"peak_date": e["peak_date"], "h": h, "observed_ratio": float(obs),
                                   "predicted_ratio": float(fit_cal["phi"] ** h)})
    dr = pd.DataFrame(decay_rows)
    mae_decay = float(np.mean(np.abs(dr["observed_ratio"] - dr["predicted_ratio"]))) if len(dr) else None
    weekly_phi = fit_cal["phi"] ** (12.0 / 52.0)
    eps_cal = episodes(cal.reset_index(drop=True))
    eps_all = episodes(df)
    res = {
        "source_sha256": sha, "n_months": int(len(df)), "first": str(df["date"].iloc[0].date()), "last": str(df["date"].iloc[-1].date()),
        "calibration": {"window": [str(cal["date"].iloc[0].date()), str(cal["date"].iloc[-1].date())], **fit_cal},
        "validation": {"window": [str(val["date"].iloc[0].date()), str(val["date"].iloc[-1].date())], **fit_val,
                       "one_step_95_coverage_with_calibration_params": cover, "n_one_step": int(len(idx)),
                       "decay_mae_h1_to_h6": mae_decay, "n_decay_points": int(len(dr))},
        "full_sample": fit_all,
        "weekly_persistence_from_calibration": float(weekly_phi),
        "weekly_half_life_weeks": float(np.log(0.5) / np.log(weekly_phi)),
        "episodes_threshold": 1.0, "episodes_calibration": eps_cal, "episodes_validation": eps_val, "episodes_all": eps_all,
        "peaks_all": [e["peak"] for e in eps_all],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/gscpi.json").write_text(json.dumps(res, indent=2))
    if len(dr):
        dr.to_csv(ROOT / "results/gscpi_decay_validation.csv", index=False)
    premise = {
        "signal": "monthly persistence of global supply-chain pressure (AR(1) coefficient, calibration window)",
        "phi": fit_cal["phi"], "phi_ci95_low": fit_cal["phi_ci95"][0], "phi_ci95_high": fit_cal["phi_ci95"][1],
        "half_life_months": fit_cal["half_life_months"], "threshold": 0.5,
    }
    (ROOT / "probe").mkdir(exist_ok=True)
    (ROOT / "probe/premise.json").write_text(json.dumps(premise, indent=2))
    print(json.dumps({k: res[k] for k in ("calibration", "validation", "weekly_persistence_from_calibration", "weekly_half_life_weeks")}, indent=2))
    print("episodes all:", json.dumps(eps_all, indent=1))


if __name__ == "__main__":
    main()
