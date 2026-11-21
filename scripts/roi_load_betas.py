#!/usr/bin/env python3
"""ROI x load analysis on Sara's existing per-subject fsaverage GLMs
(CentralAttentionManipulation, attModulated.sm5.{lh,rh}).

Reads (never writes) Sara's beta.nii.gz; frames 0-6 are the 7 ratio
conditions (.10 .30 .38 .40 .42 .44 .46), as in her Analysis/*.m code.
Data were intensity-normalized to mean 100 (inorm 100), so betas are
approximately % signal change relative to baseline fixation.

Atlases (fsaverage, 163842 vertices/hemi), in ../atlases:
  wang2015   max-prob retinotopic labels (IPS0-5, SPL1, FEF, V1-V3 ...)
  schaefer2018  400 parcels / 17 networks
  glasser2016   HCP-MMP1.0
  benson2014    template eccentricity (for fovea/periphery splits)

Outputs (to ../outputs):
  roi_betas_long.csv   subject, hemi, atlas, roi, n_vertices, ratio, beta
  roi_slopes.csv       per subject x roi (hemis averaged): slope of beta vs
                       ratio (OLS), Spearman rho, beta at .10 and .46
  group_slopes.csv     per roi: mean slope, SEM, t, p (one-sample, df=n-1),
                       FDR q within atlas; for all 8 subjects and excluding
                       S12/S13 (flat behavior)
"""
import os
import re
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

PROJ = "/projectnb/vision/CentralAttentionManipulation"
ROOT = "/projectnb/vision/CentralAttentionManipulation_reanalysis"
ATL = f"{ROOT}/atlases"
OUT = f"{ROOT}/outputs"
SUBJECTS = [1, 3, 4, 7, 9, 11, 12, 13]
FLAT_BEHAVIOR = [12, 13]
RATIOS = np.array([.10, .30, .38, .40, .42, .44, .46])
HEMIS = ["lh", "rh"]
MIN_VERTICES = 20

WANG = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "VO1", "VO2", "PHC1",
        "PHC2", "TO2", "TO1", "LO2", "LO1", "V3b", "V3a", "IPS0", "IPS1",
        "IPS2", "IPS3", "IPS4", "IPS5", "SPL1", "FEF"]
WANG_GROUPS = {
    "V1": ["V1v", "V1d"], "V2": ["V2v", "V2d"], "V3": ["V3v", "V3d"],
    "hV4": ["hV4"], "VO": ["VO1", "VO2"], "LO": ["LO1", "LO2"],
    "TO(MT+)": ["TO1", "TO2"], "V3ab": ["V3a", "V3b"],
    "IPS0": ["IPS0"], "IPS1": ["IPS1"], "IPS2": ["IPS2"], "IPS3": ["IPS3"],
    "IPS4/5": ["IPS4", "IPS5"], "SPL1": ["SPL1"], "FEF": ["FEF"],
}
# Eccentricity bands from the stimulus code: central grid 0.5-1.5 deg,
# peripheral noise annulus 3-7.5 deg (Sara's MeanBetas_SepPars_pRFeccns*.m).
ECC_BANDS = {"fovea": (0.5, 1.5), "periph": (3.0, 7.5)}
GLASSER_KEEP = ["IPS1", "IP0", "IP1", "IP2", "LIPv", "LIPd", "VIP", "MIP",
                "AIP", "7PC", "7AL", "7Am", "7PL", "7Pm", "FEF", "PEF", "6a",
                "PGp", "PFt", "V3CD", "V7", "MST", "MT"]
SCHAEFER_NETS = ["DorsAttnA", "DorsAttnB", "SalVentAttnA", "SalVentAttnB",
                 "ContA", "ContB", "ContC", "DefaultA", "DefaultB", "DefaultC",
                 "VisCent", "VisPeri"]


def load_mgz(path):
    return np.asarray(nib.load(path).dataobj).ravel()


def build_rois(hemi):
    """Return {(atlas, roi): boolean vertex mask} for one hemisphere."""
    rois = {}
    wang = load_mgz(f"{ATL}/wang2015/{hemi}.wang15_mplbl.v1_0.mgz").astype(int)
    ecc = load_mgz(f"{ATL}/benson2014/{hemi}.benson14_eccen.v4_0.mgz")
    for g, members in WANG_GROUPS.items():
        m = np.isin(wang, [WANG.index(x) + 1 for x in members])
        rois[("wang2015", g)] = m
        if g in ("V1", "V2", "V3"):
            for band, (lo, hi) in ECC_BANDS.items():
                rois[("wang2015", f"{g}_{band}")] = m & (ecc > lo) & (ecc < hi)

    lab, _, names = nib.freesurfer.read_annot(
        f"{ATL}/schaefer2018/{hemi}.Schaefer2018_400Parcels_17Networks_order.annot")
    names = [n.decode() for n in names]
    sub = {}
    for idx, n in enumerate(names):
        mt = re.match(r"17Networks_[LR]H_([A-Za-z]+)_?([A-Za-z]*)_?\d*$", n)
        if not mt:
            continue
        net, region = mt.group(1), mt.group(2) or "all"
        if net not in SCHAEFER_NETS:
            continue
        sub.setdefault((net, None), []).append(idx)
        sub.setdefault((net, region), []).append(idx)
    for (net, region), idxs in sub.items():
        name = net if region is None else f"{net}_{region}"
        rois[("schaefer2018", name)] = np.isin(lab, idxs)

    lab, _, names = nib.freesurfer.read_annot(f"{ATL}/glasser2016/{hemi}.HCP-MMP1.annot")
    names = [n.decode() for n in names]
    for area in GLASSER_KEEP:
        want = f"{hemi[0].upper()}_{area}_ROI"
        if want in names:
            rois[("glasser2016", area)] = lab == names.index(want)
    return rois


def fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum():
        ps = p[ok]
        order = np.argsort(ps)
        ranked = ps[order] * ok.sum() / (np.arange(ok.sum()) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty_like(ps)
        out[order] = np.minimum(ranked, 1)
        q[ok] = out
    return q


def main():
    os.makedirs(OUT, exist_ok=True)
    rois = {h: build_rois(h) for h in HEMIS}
    rows = []
    for s in SUBJECTS:
        for h in HEMIS:
            f = f"{PROJ}/sess0{s}/task/attModulated.sm5.{h}/beta.nii.gz"
            beta = np.asarray(nib.load(f).dataobj).reshape(163842, -1)[:, :7]
            for (atlas, roi), mask in rois[h].items():
                n = int(mask.sum())
                if n < MIN_VERTICES:
                    continue
                means = beta[mask].mean(0)
                for r, b in zip(RATIOS, means):
                    rows.append((s, h, atlas, roi, n, r, float(b)))
    long = pd.DataFrame(rows, columns=["subject", "hemi", "atlas", "roi",
                                       "n_vertices", "ratio", "beta"])
    long.to_csv(f"{OUT}/roi_betas_long.csv", index=False)

    # Average hemispheres (only ROIs present in both), then per-subject fits.
    both = long.groupby(["subject", "atlas", "roi", "ratio"]).filter(
        lambda d: d.hemi.nunique() == 2)
    bh = both.groupby(["subject", "atlas", "roi", "ratio"], as_index=False).beta.mean()
    fits = []
    for (s, atlas, roi), d in bh.groupby(["subject", "atlas", "roi"]):
        d = d.sort_values("ratio")
        slope = np.polyfit(d.ratio, d.beta, 1)[0]
        rho = stats.spearmanr(d.ratio, d.beta).statistic
        fits.append((s, atlas, roi, slope, rho,
                     d.beta.iloc[0], d.beta.iloc[-1], d.beta.mean()))
    fits = pd.DataFrame(fits, columns=["subject", "atlas", "roi", "slope_per_unit_ratio",
                                       "spearman_rho", "beta_10", "beta_46", "beta_mean"])
    fits.to_csv(f"{OUT}/roi_slopes.csv", index=False)

    grp = []
    for label, subs in [("all8", SUBJECTS),
                        ("excl12_13", [x for x in SUBJECTS if x not in FLAT_BEHAVIOR])]:
        f = fits[fits.subject.isin(subs)]
        for (atlas, roi), d in f.groupby(["atlas", "roi"]):
            sl = d.slope_per_unit_ratio.values
            t, p = stats.ttest_1samp(sl, 0)
            grp.append((label, atlas, roi, len(sl), sl.mean(), sl.std(ddof=1) / np.sqrt(len(sl)),
                        t, p, d.spearman_rho.mean(), d.beta_mean.mean()))
    grp = pd.DataFrame(grp, columns=["subjects", "atlas", "roi", "n", "mean_slope", "sem",
                                     "t", "p", "mean_rho", "mean_beta"])
    grp["q_fdr_within_atlas"] = grp.groupby(["subjects", "atlas"]).p.transform(fdr)
    grp.to_csv(f"{OUT}/group_slopes.csv", index=False)
    print(f"wrote {len(long)} rows, {len(fits)} fits, {len(grp)} group rows to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
