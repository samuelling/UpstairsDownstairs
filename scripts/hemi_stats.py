#!/usr/bin/env python3
"""Hemisphere analysis of load effects (reads outputs/roi_betas_long*.csv).

Per ROI present in both hemispheres for all subjects:
  - per-hemi OLS slope of beta vs ratio per subject; one-sample t across subjects
  - paired test of slopes rh - lh (hemisphere x load interaction)
  - per-hemi ROI-average model fits (flat/linear/exp/NR, AICc summed over subjects)
FDR within atlas, separately for lh, rh and rh-lh.
-> outputs/hemi_slopes.csv, outputs/hemi_curves.csv (group mean/SEM per hemi x ratio)
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_load_betas as base
import load_models as L

OUT = base.OUT
d = pd.concat([pd.read_csv(f"{OUT}/roi_betas_long.csv"),
               pd.read_csv(f"{OUT}/roi_betas_long_prf.csv").drop(columns="ecc_source")])
rows, curves = [], []
for (atlas, roi), g in d.groupby(["atlas", "roi"]):
    piv = g.pivot_table(index=["subject", "hemi"], columns="ratio", values="beta")
    subs = sorted(g.subject.unique())
    if len(piv) != 2 * len(base.SUBJECTS):
        continue
    Y = {h: np.vstack([piv.loc[(s, h)].values for s in base.SUBJECTS]) for h in base.HEMIS}  # subj x 7
    sl = {h: np.polyfit(base.RATIOS, Y[h].T, 1)[0] for h in base.HEMIS}
    sl["rh-lh"] = sl["rh"] - sl["lh"]
    for h in ["lh", "rh", "rh-lh"]:
        t, p = stats.ttest_1samp(sl[h], 0)
        row = dict(atlas=atlas, roi=roi, hemi=h, n=len(sl[h]), mean_slope=sl[h].mean(),
                   sem=sl[h].std(ddof=1) / np.sqrt(len(sl[h])), t=t, p=p,
                   **{f"s{s}": v for s, v in zip(base.SUBJECTS, sl[h])})
        if h != "rh-lh":
            fits = L.fit_all(Y[h].T)
            A = np.vstack([fits[m]["aicc"] for m in L.MODELS])
            sums = A.sum(1)
            row.update(best_summed=L.MODELS[int(sums.argmin())], dAICc_summed=float(sums[0] - sums[1:].min()),
                       nsubj_load=int((A.argmin(0) > 0).sum()), mean_beta=float(Y[h].mean()))
            for j, r in enumerate(base.RATIOS):
                curves.append(dict(atlas=atlas, roi=roi, hemi=h, ratio=r, m=Y[h][:, j].mean(),
                                   se=Y[h][:, j].std(ddof=1) / np.sqrt(len(Y[h]))))
        rows.append(row)
o = pd.DataFrame(rows)
o["atlas_grp"] = np.where((o.atlas == "schaefer2018") & ~o.roi.str.contains("_"), "schaeferNet", o.atlas)
o["q"] = o.groupby(["atlas_grp", "hemi"]).p.transform(base.fdr)
o.drop(columns="atlas_grp").to_csv(f"{OUT}/hemi_slopes.csv", index=False)
pd.DataFrame(curves).to_csv(f"{OUT}/hemi_curves.csv", index=False)
print(f"{len(o)} rows")

# ---- composites: average each subject's per-hemi load effect across ROI sets, test R vs L ----
SUBJ_COLS = [f"s{s}" for s in base.SUBJECTS]
COMPOSITES = [
    ("Wang dorsal (IPS0–IPS4/5, SPL1, FEF)", "wang2015", ["IPS0", "IPS1", "IPS2", "IPS3", "IPS4/5", "SPL1", "FEF"]),
    ("Wang IPS0–IPS4/5", "wang2015", ["IPS0", "IPS1", "IPS2", "IPS3", "IPS4/5"]),
    ("Schaefer dorsal attention A+B", "schaefer2018", ["DorsAttnA", "DorsAttnB"]),
    ("Schaefer control A", "schaefer2018", ["ContA"]),
    ("Schaefer default A–C", "schaefer2018", ["DefaultA", "DefaultB", "DefaultC"]),
    ("Wang V1–V3", "wang2015", ["V1", "V2", "V3"]),
]
crow = []
span = base.RATIOS[-1] - base.RATIOS[0]
for label, atlas, rois in COMPOSITES:
    x = o[(o.atlas == atlas) & o.roi.isin(rois)]
    Lf = x[x.hemi == "lh"][SUBJ_COLS].mean().values * span
    Rf = x[x.hemi == "rh"][SUBJ_COLS].mean().values * span
    Lb = x[x.hemi == "lh"].mean_beta.mean(); Rb = x[x.hemi == "rh"].mean_beta.mean()
    t, p = stats.ttest_rel(Rf, Lf); d = Rf - Lf
    crow.append(dict(label=label, atlas=atlas, rois=";".join(rois), L=Lf.mean(), R=Rf.mean(), diff=d.mean(),
                     sem=d.std(ddof=1) / np.sqrt(len(d)), t=t, p=p, wilcoxon_p=stats.wilcoxon(Rf, Lf).pvalue,
                     n_R_gt_L=int((d > 0).sum()), mean_beta_L=Lb, mean_beta_R=Rb,
                     rel_L=Lf.mean() / Lb if Lb else np.nan, rel_R=Rf.mean() / Rb if Rb else np.nan))
pd.DataFrame(crow).to_csv(f"{OUT}/hemi_composites.csv", index=False)
print(pd.DataFrame(crow)[["label", "L", "R", "diff", "t", "p", "n_R_gt_L", "rel_L", "rel_R"]].round(3).to_string(index=False))
