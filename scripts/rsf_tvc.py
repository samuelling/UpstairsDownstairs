#!/usr/bin/env python3
"""Pilot contrast-discrimination (TvC) data from the AttRSF experiment: does central
load act on the contrast response function as contrast gain or response gain?

Design (AttentionalRespFunctionExp.m): 50-ms display of red/blue squares at fixation
(majority-color task) plus two peripheral 6 cpd gratings, one at a pedestal contrast and
one at a higher test contrast; subjects report which is higher. QUEST tracks the test
contrast per blue count; threshold dC = final estimate - pedestal. Pedestal varies across
runs; each run interleaves two central loads (each with both majority colors).

Model: Naka-Rushton CRF R(c) = Rmax c^n / (c^n + c50^n); threshold dC solves
R(c + dC) - R(c) = 1 (unit discriminability), giving the dipper-shaped TvC. Fit to every
staircase's log threshold. Hypotheses about load:
  none    one CRF for all loads                          (3 parameters)
  cg      contrast gain: c50 depends on load              (+1 per extra load, or log-linear)
  rg      response gain: Rmax depends on load
  both    c50 and Rmax depend on load
Compared by AICc. Input: data/psychophysics/rsf_pilot_cells_raw.csv (from the cluster).
Output: data/psychophysics/rsf_fits.csv, rsf_curves.csv, rsf_cells.csv
"""
import os
import numpy as np, pandas as pd
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "data", "psychophysics")
D = pd.read_csv(os.path.join(P, "rsf_pilot_cells_raw.csv"))
# sets: A-100 / B-100 (loads 5% vs 48%, 7 / 11 pedestals), A-200 (6 loads x 7 pedestals),
# B-200 (10 loads at 2 pedestals, plus the hardest loads at 3 more); subjects coded on the cluster
D = D[(D.thr + D.ped) < 0.98]                                   # drop staircases pinned at the 0.99 ceiling
D["lthr"] = np.log10(D.thr)


def tvc(c, rmax, c50, n):
    r0 = rmax * c ** n / (c ** n + c50 ** n)
    T = r0 + 1.0
    out = np.full_like(c, np.inf, dtype=float)
    ok = T < rmax * 0.999
    out[ok] = c50 * (T[ok] / (rmax - T[ok])) ** (1 / n) - c[ok]
    return out


def fit_set(d, hyp, loadfx):
    """loadfx: 'free' (one parameter per load) or 'linear' (log-param linear in load)."""
    loads = np.sort(d.ratio.unique()); li = np.searchsorted(loads, d.ratio.values)
    z = (loads - loads.mean()) / (loads.std() if loads.std() > 0 else 1)
    c, y = d.ped.values, d.lthr.values
    nL = len(loads)
    k_extra = 0 if hyp == "none" else ((nL - 1) if loadfx == "free" else 1) * (2 if hyp == "both" else 1)

    def unpack(p):
        lr, lc, ln = p[:3]; ex = p[3:]
        lR = np.full(nL, lr); lC = np.full(nL, lc)
        if hyp != "none":
            parts = np.split(ex, 2) if hyp == "both" else [ex]
            for which, e in zip(["cg", "rg"] if hyp == "both" else [hyp], parts):
                delta = np.r_[0, e] if loadfx == "free" else e[0] * z
                if which == "cg": lC = lC + delta
                else: lR = lR + delta
        return lR, lC, np.exp(ln)

    def resid(p):
        lR, lC, n = unpack(p)
        pred = tvc(c, np.exp(lR[li]), np.exp(lC[li]), n)
        r = y - np.log10(np.where(np.isfinite(pred) & (pred > 0), pred, 10.0))
        return r

    best = None
    for lr0 in [np.log(5), np.log(20), np.log(60)]:
        for lc0 in [np.log(0.02), np.log(0.08)]:
            for ln0 in [np.log(1.5), np.log(2.5)]:
                p0 = np.r_[lr0, lc0, ln0, np.zeros(k_extra)]
                try:
                    res = optimize.least_squares(resid, p0, bounds=(np.r_[np.log(1.2), np.log(1e-3), np.log(0.5), -3 * np.ones(k_extra)],
                                                                   np.r_[np.log(1e4), np.log(2), np.log(6), 3 * np.ones(k_extra)]))
                except ValueError:
                    continue
                if best is None or res.cost < best.cost:
                    best = res
    rss = 2 * best.cost; N = len(y); k = 3 + k_extra
    aicc = N * np.log(rss / N) + 2 * k + 2 * k * (k + 1) / (N - k - 1)
    lR, lC, n = unpack(best.x)
    return dict(aicc=aicc, rss=rss, N=N, k=k, loads=loads, rmax=np.exp(lR), c50=np.exp(lC), n=n)


fits, curves = [], []
cgrid = np.logspace(np.log10(0.004), np.log10(0.65), 80)
for s, d in D.groupby("set"):
    loadfx = "free" if d.ratio.nunique() == 2 else "linear"
    res = {h: fit_set(d, h, loadfx) for h in ["none", "cg", "rg", "both"]}
    amin = min(r["aicc"] for r in res.values())
    for h, r in res.items():
        for L, rm, c5 in zip(r["loads"], r["rmax"], r["c50"]):
            fits.append(dict(set=s, hyp=h, loadfx=loadfx, aicc=r["aicc"], dAICc=r["aicc"] - amin, N=r["N"], k=r["k"], load=L, rmax=rm, c50=c5, n=r["n"]))
        for L, rm, c5 in zip(r["loads"], r["rmax"], r["c50"]):
            t = tvc(cgrid, rm, c5, r["n"])
            for cc, tt in zip(cgrid, t):
                if np.isfinite(tt) and tt > 0:
                    curves.append(dict(set=s, hyp=h, load=L, ped=cc, thr=tt))
F = pd.DataFrame(fits); F.to_csv(os.path.join(P, "rsf_fits.csv"), index=False)
pd.DataFrame(curves).to_csv(os.path.join(P, "rsf_curves.csv"), index=False)
# cell summaries (geometric mean +- SEM in log units) for plotting
cells = D.groupby(["set", "ped", "ratio"]).agg(gm=("lthr", "mean"), sd=("lthr", "std"), n=("lthr", "size"), rg_acc=("rg_acc", "mean")).reset_index()
cells["se"] = cells.sd / np.sqrt(cells.n); cells["thr"] = 10 ** cells.gm; cells["lo"] = 10 ** (cells.gm - cells.se); cells["hi"] = 10 ** (cells.gm + cells.se)
cells.drop(columns=["sd"]).to_csv(os.path.join(P, "rsf_cells.csv"), index=False)

pd.set_option("display.width", 220)
summ = F.groupby(["set", "hyp"]).agg(dAICc=("dAICc", "first"), k=("k", "first"), N=("N", "first"), n=("n", "first")).reset_index()
print(summ.round(2).to_string(index=False))
print(F[F.hyp.isin(["cg", "rg", "both"])].groupby(["set", "hyp", "load"])[["c50", "rmax"]].first().round(4).to_string())
