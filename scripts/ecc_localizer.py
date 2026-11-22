#!/usr/bin/env python3
"""Where in early visual cortex does central load raise or lower responses? (cluster)

(2) Eccentricity profile. For V1, V2, V3 (each subject's pRF-defined areas, pRF R2 > 10%;
    S9 has no pRF: Wang areas + Benson template eccentricity), vertices are binned by
    pRF eccentricity from 0.25 to 14 deg. In each bin the 7 load betas (original FS-FAST
    GLMs, hemispheres pooled by vertex) are averaged and the load slope is fit; reported
    as the fitted change from 10% to 46% load (% signal). Group mean +- SEM, t-test per bin.
(3) Annulus localizer. Every subject ran 2 localizer runs in which the same peripheral
    noise annulus alternated with fixation (On-vs-null). Within V1-V3, vertices are split
    by their localizer response: annulus-driven (sig > 3, i.e. p < .001, positive) vs not
    driven (|sig| < 1), and for driven vertices also by localizer strength (terciles).
    Load slopes are computed as above.

Reads the original study and the reanalysis atlases/pRF maps (read-only); writes
outputs/ecc_profile.csv, outputs/localizer_rois.csv.
"""
import os, sys
import numpy as np, pandas as pd, nibabel as nib
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_load_betas as base

PROJ, ROOT, OUT = base.PROJ, base.ROOT, base.OUT
PRF = f"{ROOT}/prf_fsaverage"
NV = 163842
R = base.RATIOS
EDGES = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0, 14.0])
WANG = base.WANG
mgz = lambda p: np.asarray(nib.load(p).dataobj).ravel()


def areas(s, h):
    """{area: (mask, ecc)} for V1-V3 in subject s, hemisphere h."""
    out = {}
    sid = f"{s:03d}"
    if os.path.exists(f"{PRF}/{h}.S{sid}.pRF_ecc.fsaverage.mgz"):
        ecc = mgz(f"{PRF}/{h}.S{sid}.pRF_ecc.fsaverage.mgz"); r2 = mgz(f"{PRF}/{h}.S{sid}.pRF_R2.fsaverage.mgz")
        for a in ["v1", "v2", "v3"]:
            m = np.zeros(NV, bool)
            m[nib.freesurfer.read_label(f"{PRF}/{h}.S{sid}.pRF.{a}.fsaverage.label")] = True
            out[a.upper()] = (m & (r2 > 10), ecc, "pRF")
    else:
        wang = mgz(f"{ROOT}/atlases/wang2015/{h}.wang15_mplbl.v1_0.mgz").astype(int)
        ecc = mgz(f"{ROOT}/atlases/benson2014/{h}.benson14_eccen.v4_0.mgz")
        for a, mem in [("V1", ["V1v", "V1d"]), ("V2", ["V2v", "V2d"]), ("V3", ["V3v", "V3d"])]:
            out[a] = (np.isin(wang, [WANG.index(x) + 1 for x in mem]), ecc, "template")
    return out


def slope(b):  # b: (n_vertices, 7) -> fitted change 10% -> 46% of the ROI-mean curve
    y = b.mean(0)
    return np.polyfit(R, y, 1)[0] * (R[-1] - R[0]), y.mean()


ecc_rows, loc_rows = [], []
for s in base.SUBJECTS:
    per = {}
    for h in base.HEMIS:
        beta = np.asarray(nib.load(f"{PROJ}/sess0{s}/task/attModulated.sm5.{h}/beta.nii.gz").dataobj).reshape(NV, -1)[:, :7]
        sig = mgz(f"{PROJ}/sess0{s}/localizer/localizer.sm5.{h}/On-vs-null/sig.nii.gz")
        for a, (m, ecc, src) in areas(s, h).items():
            d = per.setdefault(a, {"beta": [], "ecc": [], "sig": [], "src": src})
            d["beta"].append(beta[m]); d["ecc"].append(ecc[m]); d["sig"].append(sig[m])
    for a, d in per.items():
        b = np.vstack(d["beta"]); e = np.concatenate(d["ecc"]); g = np.concatenate(d["sig"])
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            k = (e >= lo) & (e < hi)
            if k.sum() >= 10:
                sl, mb = slope(b[k])
                ecc_rows.append(dict(subject=s, area=a, src=d["src"], ecc_lo=lo, ecc_hi=hi, ecc_mid=np.sqrt(lo * hi), n=int(k.sum()), effect=sl, mean_beta=mb,
                                     loc_sig=float(np.median(g[k]))))
        sets = {"annulus-driven (localizer p < .001)": g > 3, "not driven (|sig| < 1)": np.abs(g) < 1,
                "annulus-driven, 3-7.5 deg": (g > 3) & (e >= 3) & (e <= 7.5), "3-7.5 deg band (pRF only)": (e >= 3) & (e <= 7.5),
                "not driven, 3-7.5 deg": (np.abs(g) < 1) & (e >= 3) & (e <= 7.5)}
        q = np.quantile(g[g > 3], [1 / 3, 2 / 3]) if (g > 3).sum() > 30 else None
        if q is not None:
            sets["driven, weakest third"] = (g > 3) & (g <= q[0]); sets["driven, middle third"] = (g > q[0]) & (g <= q[1]); sets["driven, strongest third"] = g > q[1]
        for name, k in sets.items():
            if k.sum() >= 20:
                sl, mb = slope(b[k])
                loc_rows.append(dict(subject=s, area=a, roi=name, n=int(k.sum()), effect=sl, mean_beta=mb, median_ecc=float(np.median(e[k]))))
E = pd.DataFrame(ecc_rows); Lc = pd.DataFrame(loc_rows)


def group(df, keys):
    out = []
    for kv, x in df.groupby(keys, sort=False):
        t, p = stats.ttest_1samp(x.effect, 0) if len(x) > 2 else (np.nan, np.nan)
        out.append(dict(zip(keys, kv if isinstance(kv, tuple) else (kv,)), n_subj=len(x), mean=x.effect.mean(), sem=x.effect.std(ddof=1) / np.sqrt(len(x)), t=t, p=p,
                        n_neg=int((x.effect < 0).sum()), mean_beta=x.mean_beta.mean(), n_vert=int(x.n.median())))
    return pd.DataFrame(out)


GE = group(E, ["area", "ecc_lo", "ecc_hi", "ecc_mid"])
GL = group(Lc, ["area", "roi"])
# pooled V1-V3: average each subject's three areas first
Ep = E.groupby(["subject", "ecc_lo", "ecc_hi", "ecc_mid"], as_index=False).agg(effect=("effect", "mean"), mean_beta=("mean_beta", "mean"), n=("n", "sum"))
Lp = Lc.groupby(["subject", "roi"], as_index=False).agg(effect=("effect", "mean"), mean_beta=("mean_beta", "mean"), n=("n", "sum"))
GE = pd.concat([GE, group(Ep.assign(area="V1–V3"), ["area", "ecc_lo", "ecc_hi", "ecc_mid"])])
GL = pd.concat([GL, group(Lp.assign(area="V1–V3"), ["area", "roi"])])
pd.concat([E.assign(kind="subject"), GE.assign(kind="group")]).to_csv(f"{OUT}/ecc_profile.csv", index=False)
pd.concat([Lc.assign(kind="subject"), GL.assign(kind="group")]).to_csv(f"{OUT}/localizer_rois.csv", index=False)
pd.set_option("display.width", 220)
print(GE[GE.area == "V1–V3"].round(3).to_string(index=False))
print(GL.sort_values(["area", "roi"]).round(3).to_string(index=False))
