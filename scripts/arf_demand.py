#!/usr/bin/env python3
"""Is the attentional response function (ARF) compressive once load is put on a
demand axis? Runs locally on data/ (stdlib + numpy/scipy).

The load-function models (load_models.py) found ROI responses linear in the
minority ratio. But ratio is not a linear measure of demand: behavior is at ceiling
from 10% to 30% and collapses from 38% to 46%, and the evidence for the majority
(s = 0.5 - ratio) shrinks 10-fold. Here each ROI's response is refit against three
x axes, each rescaled to 0-1:
  ratio      the design variable (as before)
  behavior   group-mean behavioral demand = 1 - task score (misses count as errors)
  evidence   1/s, the number of samples needed scales with 1/s^2; we use log(0.4/s),
             a contrast-like (log) axis, from 0 at 10% to 1 at 46%
and fit a power function  b = a + k * x^q  (q < 1 compressive, q = 1 linear,
q > 1 expansive; q grid-searched, a,k by least squares), plus a flat and a linear
model; models are compared with AICc. Per subject (hemispheres averaged) and on
the group-mean curve.

Output: data/arf_demand.csv
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
D = json.loads((s := open(os.path.join(HERE, "..", "dashboard", "data.js")).read())[s.index("{"):s.rindex("}") + 1])
RAT = np.array(D["ratios"])
score = np.array([D["behavior"]["score"]["subjects"][str(x)] for x in D["subjects"]])  # % correct of all flips
demand = 1 - np.maximum.accumulate(score.mean(0)[::-1])[::-1] / 100  # group mean, monotone non-decreasing
AXES = {
    "ratio": (RAT - RAT[0]) / (RAT[-1] - RAT[0]),
    "behavior": (demand - demand[0]) / (demand[-1] - demand[0]),
    "evidence": np.log(0.4 / (0.5 - RAT)) / np.log(0.4 / 0.04),
}
QS = np.exp(np.linspace(np.log(0.1), np.log(10), 81))
N = len(RAT)


def aicc(rss, k):
    return N * np.log(max(rss, 1e-12) / N) + 2 * k + 2 * k * (k + 1) / (N - k - 1)


def fit(y, x):
    out = {}
    out["flat"] = aicc(((y - y.mean()) ** 2).sum(), 1)
    X = np.c_[np.ones(N), x]; r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]; out["linear"] = aicc((r ** 2).sum(), 2)
    best = (np.inf, 1.0)
    for q in QS:
        X = np.c_[np.ones(N), x ** q]; r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        best = min(best, ((r ** 2).sum(), q))
    out["power"] = aicc(best[0], 3); out["q"] = best[1]
    return out


ROIS = {"IPS0": "IPS0", "IPS1": "IPS1", "IPS2": "IPS2", "IPS3": "IPS3", "IPS4/5": "IPS4/5", "SPL1": "SPL1", "FEF": "FEF", "LO": "LO"}
L = pd.read_csv(os.path.join(DATA, "roi_betas_long.csv"))
W = L[L.atlas == "wang2015"].groupby(["subject", "roi", "ratio"], as_index=False).beta.mean()
curves = {}
for name, roi in ROIS.items():
    curves[name] = W[W.roi == roi].pivot(index="subject", columns="ratio", values="beta").values
curves["Dorsal composite"] = np.mean([curves[r] for r in ["IPS0", "IPS1", "IPS2", "IPS3", "IPS4/5", "SPL1", "FEF"]], axis=0)
curves["Foveal V1–V3"] = np.mean([W[W.roi == f"{a}_fovea"].pivot(index="subject", columns="ratio", values="beta").values for a in ["V1", "V2", "V3"]], axis=0)

rows = []
for name, Y in curves.items():
    for ax, x in AXES.items():
        per = [fit(y, x) for y in Y]
        q = np.array([p["q"] for p in per]); lq = np.log(q)
        t, p = stats.ttest_1samp(lq, 0)
        g = fit(Y.mean(0), x)
        rows.append(dict(roi=name, axis=ax, q_group_curve=g["q"], dAICc_linear_minus_power_group=g["linear"] - g["power"],
                         q_subject_geomean=float(np.exp(lq.mean())), n_compressive=int((q < 1).sum()), n=len(q),
                         t_logq=t, p_logq=p,
                         n_power_beats_linear=int(sum(pp["linear"] - pp["power"] > 2 for pp in per)),
                         sum_dAICc_flat_minus_best=float(sum(pp["flat"] - min(pp["linear"], pp["power"]) for pp in per))))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(DATA, "arf_demand.csv"), index=False)
pd.DataFrame({"ratio": RAT, "demand_behavior": demand, **{f"x_{k}": v for k, v in AXES.items()}}).to_csv(os.path.join(DATA, "arf_axes.csv"), index=False)
pd.set_option("display.width", 220)
print("demand axis (group):", np.round(demand, 3), "\naxes:", {k: np.round(v, 2).tolist() for k, v in AXES.items()})
print(R.round(3).to_string(index=False))
