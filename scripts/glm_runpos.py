#!/usr/bin/env python3
"""Run-position control for the CAM load effect (cluster batch job).

The original design has one load level per run and a fixed, repeating ratio
order, so load is partly confounded with run position. This script re-estimates
the task response run by run, then separates load from run position.

Level 1, per run x ROI (hemispheres separately): average the surface-sampled,
smoothed time series (fmcpr.siemens.sm5.fsaverage.{lh,rh}) within each ROI after
intensity-normalizing the run to a global mean of 100 (as FS-FAST inorm), then fit
  y = b_task * (16-s ON blocks convolved with the SPM canonical HRF) + quadratic drift
(the original analysis's settings: SPM HRF, polyfit 2). A second fit adds the six
motion parameters (mcprextreg). b_task is ~ % signal change vs fixation.

Level 2, per subject x ROI, regress the run betas on
  A: load only (should reproduce the original slopes)
  B: load + run position (run number within subject)
  C: load + scan-day fixed effects + run position within day
and t-test the load slopes across subjects (FDR within atlas).

Reads the original study read-only; writes to $ROOT/glm_runpos/.
"""
import glob, os, re, sys
import numpy as np, pandas as pd, nibabel as nib, scipy.io as sio
from scipy import stats
from scipy.stats import gamma

sys.path.insert(0, "/projectnb/vision/CentralAttentionManipulation_reanalysis/code")
import roi_load_betas as base   # ROI definitions, paths, FDR

PROJ, ROOT = base.PROJ, base.ROOT
OUT = f"{ROOT}/glm_runpos"
os.makedirs(OUT, exist_ok=True)
TR, NV = 2.0, 163842
LABEL = {"10Perc": .10, "30Perc": .30, "38Perc": .38, "40Perc": .40, "42Perc": .42, "44Perc": .44, "46Perc": .46}
SPAN = 0.36


def spm_hrf(dt):
    t = np.arange(0, 32, dt)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / h.sum()


def design(par, nt):
    dt = 0.05
    hi = np.zeros(int(round(nt * TR / dt)))
    ratio = None
    for line in open(par):
        p = line.split()
        if len(p) >= 5 and p[1] != "0":
            on, dur = float(p[0]), float(p[2]); ratio = LABEL[p[4]]
            hi[int(on / dt):int((on + dur) / dt)] = 1
    x = np.convolve(hi, spm_hrf(dt))[:len(hi)]
    x = x[(np.arange(nt) * TR / dt).astype(int)]
    x = x / x.max()                                        # plateau = 1 -> beta in data units
    tt = np.linspace(-1, 1, nt)
    drift = np.c_[np.ones(nt), tt, 1.5 * tt ** 2 - 0.5]     # Legendre 0-2
    return x, drift, ratio


def motion(rdir, nt):
    f = f"{rdir}/mcprextreg"
    if not os.path.exists(f):
        return None
    m = np.loadtxt(f)
    m = m.reshape(nt, -1) if m.size % nt == 0 else None
    return None if m is None else (m - m.mean(0)) / (m.std(0) + 1e-9)


def beh_time(s, rn):
    f = f"{PROJ}/Behavioral/{s}/Data_CentralAttMod_Run{rn}_{s}.mat"
    try:
        t = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["TheData"].t
        return str(t.theDate), str(t.timeStamp)
    except Exception:
        return None, None


def main():
    rois = {h: base.build_rois(h) for h in base.HEMIS}
    keys = sorted(set(rois["lh"]) & set(rois["rh"]))
    rows = []
    for s in base.SUBJECTS:
        for rdir in sorted(glob.glob(f"{PROJ}/sess0{s}/task/0[0-9][0-9]")):
            rn = int(os.path.basename(rdir))
            date, hhmm = beh_time(s, rn)
            for h in base.HEMIS:
                f = f"{rdir}/fmcpr.siemens.sm5.fsaverage.{h}.nii.gz"
                Y = np.asarray(nib.load(f).dataobj, dtype=np.float32).reshape(NV, -1)
                nt = Y.shape[1]
                vm = Y.mean(1)
                Y *= 100.0 / vm[vm > 0.1 * vm.max()].mean()        # global inorm to 100
                x, drift, ratio = design(f"{rdir}/CAM.par", nt)
                X0 = np.c_[x, drift]
                mc = motion(rdir, nt)
                X1 = np.c_[X0, mc] if mc is not None else None
                for key in keys:
                    m = rois[h][key]
                    if m.sum() < base.MIN_VERTICES:
                        continue
                    y = Y[m].mean(0)
                    b0 = np.linalg.lstsq(X0, y, rcond=None)[0][0]
                    b1 = np.linalg.lstsq(X1, y, rcond=None)[0][0] if X1 is not None else np.nan
                    rows.append((s, rn, date, hhmm, ratio, h, key[0], key[1], b0, b1))
            print(f"S{s} run {rn} ratio {ratio} date {date} {hhmm}", flush=True)
    R = pd.DataFrame(rows, columns=["subject", "run", "date", "time", "ratio", "hemi", "atlas", "roi", "beta", "beta_mc"])
    R.to_csv(f"{OUT}/run_betas.csv", index=False)
    level2(R)


def fit(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def level2(R):
    B = R.groupby(["subject", "run", "date", "time", "ratio", "atlas", "roi"], as_index=False)[["beta", "beta_mc"]].mean()
    subj = []
    for (s, atlas, roi), d in B.groupby(["subject", "atlas", "roi"]):
        d = d.sort_values("run")
        r = d.ratio.values; pos = (d.run.values - d.run.mean()) / max(1, d.run.std())
        days = pd.factorize(d.date.fillna("na"))[0]
        within = d.groupby("date", dropna=False).cumcount().values.astype(float)
        D = np.eye(days.max() + 1)[days]
        out = {"subject": s, "atlas": atlas, "roi": roi, "n_runs": len(d), "n_days": days.max() + 1,
               "corr_ratio_run": np.corrcoef(r, d.run)[0, 1]}
        for col, tag in [("beta", ""), ("beta_mc", "_mc")]:
            y = d[col].values
            if np.isnan(y).any():
                continue
            out["slopeA" + tag] = fit(y, np.c_[np.ones(len(y)), r])[1]
            bB = fit(y, np.c_[np.ones(len(y)), r, pos]); out["slopeB" + tag] = bB[1]; out["runB" + tag] = bB[2]
            XC = np.c_[D, r, within - within.mean()] if days.max() > 0 else np.c_[np.ones(len(y)), r, within - within.mean()]
            bC = fit(y, XC); out["slopeC" + tag] = bC[-2]; out["withinC" + tag] = bC[-1]
        subj.append(out)
    S = pd.DataFrame(subj)
    S.to_csv(f"{OUT}/subject_slopes.csv", index=False)
    grp = []
    for (atlas, roi), d in S.groupby(["atlas", "roi"]):
        for k in [c for c in S.columns if c.startswith(("slope", "run", "within"))]:
            v = d[k].dropna().values
            if len(v) < 3:
                continue
            t, p = stats.ttest_1samp(v, 0)
            grp.append(dict(atlas=atlas, roi=roi, measure=k, n=len(v), mean=v.mean(), effect_10_46=v.mean() * SPAN if k.startswith("slope") else np.nan,
                            sem=v.std(ddof=1) / np.sqrt(len(v)), t=t, p=p))
    G = pd.DataFrame(grp)
    G["q"] = G.groupby(["atlas", "measure"]).p.transform(base.fdr)
    G.to_csv(f"{OUT}/group_slopes_runpos.csv", index=False)
    show = G[(G.atlas == "wang2015") & G.measure.isin(["slopeA", "slopeB", "slopeC", "slopeC_mc", "runB"])]
    print(show.pivot_table(index="roi", columns="measure", values="t").round(2).to_string())
    print(S.groupby("subject")[["corr_ratio_run", "n_runs", "n_days"]].first().round(2).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "level2":
        level2(pd.read_csv(f"{OUT}/run_betas.csv", dtype={"date": str, "time": str}))
    else:
        main()
