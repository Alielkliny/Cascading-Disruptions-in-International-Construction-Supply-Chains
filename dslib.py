"""Parse DSLIB project files (Batselier and Vanhoucke, 2015; DSLIB v3.4) into activity networks.

Units: working days (8 h per day, 5 days per week), as in the DSLIB agenda sheets.
Relations: FS, SS, FF, SF with lags in w (5 working days), d or h.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

HOURS_PER_DAY = 8.0
DAYS_PER_WEEK = 5.0

_DUR_RE = re.compile(r"(?:(\d+(?:[.,]\d+)?)\s*w)?\s*(?:(\d+(?:[.,]\d+)?)\s*d(?:ays)?)?\s*(?:(\d+(?:[.,]\d+)?)\s*h)?\s*$")
_REL_RE = re.compile(r"^\s*(\d+)\s*(FS|SS|FF|SF)?\s*([+-].*)?$")


def _num(s: str | None) -> float:
    if s is None:
        return 0.0
    return float(s.replace(",", "."))


def parse_duration(value) -> float:
    """Return a duration or lag in working days."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip().lower().replace("days", "d").replace("day", "d")
    if s in ("", "0"):
        return 0.0
    s = s.replace(" ", "")
    m = re.fullmatch(r"(?:(\d+(?:[.,]\d+)?)w)?(?:(\d+(?:[.,]\d+)?)d)?(?:(\d+(?:[.,]\d+)?)h)?", s)
    if not m or not any(m.groups()):
        try:
            return float(s.replace(",", "."))
        except ValueError as exc:
            raise ValueError(f"unparsed duration {value!r}") from exc
    w, d, h = (_num(g) for g in m.groups())
    return w * DAYS_PER_WEEK + d + h / HOURS_PER_DAY


def parse_relations(text) -> list[tuple[int, str, float]]:
    """'58FS;53FS+2d' -> [(58,'FS',0.0),(53,'FS',2.0)]."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    out = []
    for tok in str(text).split(";"):
        tok = tok.strip()
        if not tok:
            continue
        m = _REL_RE.match(tok)
        if not m:
            raise ValueError(f"unparsed relation {tok!r}")
        pid = int(m.group(1))
        typ = m.group(2) or "FS"
        lag = 0.0
        if m.group(3):
            sign = -1.0 if m.group(3).strip().startswith("-") else 1.0
            lag = sign * parse_duration(m.group(3).strip()[1:])
        out.append((pid, typ, lag))
    return out


@dataclass
class Project:
    code: str
    name: str
    sector: str
    ids: np.ndarray            # activity ids (leaf activities)
    names: list
    dur: np.ndarray            # baseline durations, working days
    fixed_cost: np.ndarray     # baseline fixed cost per activity
    cost_per_hour: np.ndarray  # resource cost rate per hour (DSLIB 'Cost/Hour')
    total_cost: np.ndarray
    preds: list                # per activity: list of (pred_index, type, lag)
    order: np.ndarray          # topological order
    risk: np.ndarray           # (n,3) optimistic, most probable, pessimistic in % of duration
    pd_sheet: float            # PD (days) from the analysis sheet
    bac: float
    start_date: pd.Timestamp | None = None
    es_sheet: np.ndarray | None = None  # sheet baseline start, working days from project start
    actual_dur: np.ndarray | None = None
    actual_start: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Calendar:
    hours: list            # working hour slots as (start_hour, end_hour)
    weekmask: str          # numpy busday weekmask, e.g. '1111100'
    holidays: list         # numpy datetime64[D]

    @classmethod
    def from_agenda(cls, path: str) -> "Calendar":
        try:
            a = pd.read_excel(path, "Agenda", header=None)
        except ValueError:
            return cls([(8, 12), (13, 17)], "1111100", [])
        hours = []
        for slot, flag in zip(a[0].astype(str), a[1].astype(str)):
            m = re.match(r"(\d+):\d+\s*-\s*(\d+):\d+", slot)
            if m and flag.strip().lower() == "yes":
                hours.append((int(m.group(1)), int(m.group(2))))
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        flags = dict(zip(a[3].astype(str), a[4].astype(str)))
        mask = "".join("1" if flags.get(d, "No").strip().lower() == "yes" else "0" for d in days)
        hol = []
        for v in a[6].tolist()[1:]:
            t = pd.to_datetime(v, errors="coerce")
            if not pd.isna(t):
                hol.append(np.datetime64(t.date(), "D"))
        if not hours:
            hours = [(8, 12), (13, 17)]
        if "1" not in mask:
            mask = "1111100"
        return cls(hours, mask, hol)

    @property
    def hours_per_day(self) -> float:
        return float(sum(e - s for s, e in self.hours))

    def day_fraction(self, t: pd.Timestamp) -> float:
        h = t.hour + t.minute / 60.0
        worked = sum(min(max(h - s, 0.0), e - s) for s, e in self.hours)
        return worked / self.hours_per_day

    def working_days_between(self, t0: pd.Timestamp, t1: pd.Timestamp) -> float:
        if pd.isna(t0) or pd.isna(t1):
            return np.nan
        days = float(np.busday_count(np.datetime64(t0.date(), "D"), np.datetime64(t1.date(), "D"),
                                     weekmask=self.weekmask, holidays=self.holidays))
        return days + self.day_fraction(t1) - self.day_fraction(t0)


def _working_days_between(t0: pd.Timestamp, t1: pd.Timestamp, cal: "Calendar | None" = None) -> float:
    cal = cal or Calendar([(8, 12), (13, 17)], "1111100", [])
    return cal.working_days_between(t0, t1)


def topo_order(n: int, preds: list) -> np.ndarray:
    indeg = np.zeros(n, dtype=int)
    succ = [[] for _ in range(n)]
    for j in range(n):
        for (i, _t, _l) in preds[j]:
            indeg[j] += 1
            succ[i].append(j)
    stack = [j for j in range(n) if indeg[j] == 0]
    order = []
    while stack:
        i = stack.pop()
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                stack.append(j)
    if len(order) != n:
        raise ValueError("network has a cycle")
    return np.array(order, dtype=int)


def cpm_forward(dur: np.ndarray, preds: list, order: np.ndarray, release: np.ndarray | None = None):
    """Earliest start and finish with FS/SS/FF/SF relations and lags; start >= 0."""
    n = len(dur)
    es = np.zeros(n)
    ef = np.zeros(n)
    for j in order:
        s = 0.0 if release is None else release[j]
        for (i, t, lag) in preds[j]:
            if t == "FS":
                s = max(s, ef[i] + lag)
            elif t == "SS":
                s = max(s, es[i] + lag)
            elif t == "FF":
                s = max(s, ef[i] + lag - dur[j])
            elif t == "SF":
                s = max(s, es[i] + lag - dur[j])
        es[j] = s
        ef[j] = s + dur[j]
    return es, ef


def cpm_backward(dur, preds, order, horizon):
    n = len(dur)
    lf = np.full(n, horizon, dtype=float)
    ls = lf - dur
    succ = [[] for _ in range(n)]
    for j in range(n):
        for (i, t, lag) in preds[j]:
            succ[i].append((j, t, lag))
    for i in order[::-1]:
        f = horizon
        for (j, t, lag) in succ[i]:
            if t == "FS":
                f = min(f, ls[j] - lag)
            elif t == "SS":
                f = min(f, ls[j] - lag + dur[i])
            elif t == "FF":
                f = min(f, lf[j] - lag)
            elif t == "SF":
                f = min(f, lf[j] - lag + dur[i])
        lf[i] = f
        ls[i] = f - dur[i]
    return ls, lf


def load_project(path: str, sheet_row: dict | None = None, with_tracking: bool = True) -> Project:
    cal = Calendar.from_agenda(path)
    raw = pd.read_excel(path, "Baseline Schedule", header=1)
    raw = raw[pd.to_numeric(raw["ID"], errors="coerce").notna()].copy()
    raw["ID"] = raw["ID"].astype(int)
    wbs = raw["WBS"].astype(str).tolist()
    wbs_set = set(wbs)
    is_parent = [any(w2.startswith(w + ".") for w2 in wbs_set) for w in wbs]
    leaf = raw[~np.array(is_parent)].copy()
    id_to_idx = {int(a): k for k, a in enumerate(leaf["ID"].tolist())}
    n = len(leaf)
    dur = np.array([parse_duration(v) for v in leaf["Duration"]], dtype=float)
    preds = []
    for txt in leaf["Predecessors"].tolist():
        rels = []
        for (pid, t, lag) in parse_relations(txt):
            if pid in id_to_idx:
                rels.append((id_to_idx[pid], t, lag))
        preds.append(rels)
    order = topo_order(n, preds)

    def col(name):
        return pd.to_numeric(leaf.get(name), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    fixed = col("Fixed Cost")
    cph = col("Cost/Hour")
    tot = col("Total Cost")
    bstart = pd.to_datetime(leaf["Baseline Start"], errors="coerce")
    t0 = bstart.min()
    es_sheet = np.array([cal.working_days_between(t0, t) for t in bstart])

    # risk analysis profile (percent of baseline duration)
    risk = np.tile(np.array([100.0, 100.0, 100.0]), (n, 1))
    try:
        ra = pd.read_excel(path, "Risk Analysis", header=1)
        ra = ra[pd.to_numeric(ra["ID"], errors="coerce").notna()].copy()
        ra["ID"] = ra["ID"].astype(int)
        cols = [c for c in ra.columns if str(c).startswith("Optimistic (%)") or str(c).startswith("Most probable (%)") or str(c).startswith("Pessimistic (%)")]
        if len(cols) >= 3:
            m = ra.set_index("ID")[cols[:3]].apply(pd.to_numeric, errors="coerce")
            for a, k in id_to_idx.items():
                if a in m.index and not m.loc[a].isna().any():
                    risk[k] = m.loc[a].to_numpy(dtype=float)
    except (ValueError, KeyError):
        pass

    actual_dur = actual_start = None
    if with_tracking:
        try:
            xl = pd.ExcelFile(path)
            tps = [s for s in xl.sheet_names if s.startswith("TP") or s.startswith("Project Control - TP")]
            if tps:
                last = tps[-1]
                tp = pd.read_excel(xl, last, header=3)
                tp = tp[pd.to_numeric(tp["ID"], errors="coerce").notna()].copy()
                tp["ID"] = tp["ID"].astype(int)
                tp = tp.set_index("ID")
                ad = np.full(n, np.nan)
                ast = np.full(n, np.nan)
                for a, k in id_to_idx.items():
                    if a in tp.index:
                        v = tp.loc[a, "Actual Duration"]
                        if isinstance(v, pd.Series):
                            v = v.iloc[0]
                        try:
                            ad[k] = parse_duration(v)
                        except ValueError:
                            ad[k] = np.nan
                        s = pd.to_datetime(tp.loc[a, "Actual Start"], errors="coerce")
                        if isinstance(s, pd.Series):
                            s = s.iloc[0]
                        ast[k] = cal.working_days_between(t0, s) if not pd.isna(s) else np.nan
                actual_dur, actual_start = ad, ast
        except (ValueError, KeyError):
            pass

    sr = sheet_row or {}
    return Project(
        code=str(sr.get("Code", "")), name=str(sr.get("Project name", "")), sector=str(sr.get("Sector", "")),
        ids=leaf["ID"].to_numpy(dtype=int), names=leaf["Name"].astype(str).tolist(), dur=dur,
        fixed_cost=fixed, cost_per_hour=cph, total_cost=tot, preds=preds, order=order, risk=risk,
        pd_sheet=float(pd.to_numeric(sr.get("PD (days)", np.nan), errors="coerce")),
        bac=float(pd.to_numeric(str(sr.get("BAC", "nan")).replace("€", "").replace(" ", "").replace(",", "."), errors="coerce")),
        start_date=t0, es_sheet=es_sheet, actual_dur=actual_dur, actual_start=actual_start,
        meta={"hours_per_day": cal.hours_per_day, "weekmask": cal.weekmask, "n_holidays": len(cal.holidays)},
    )
