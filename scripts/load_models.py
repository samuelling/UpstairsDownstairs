#!/usr/bin/env python3
"""Load-function model comparison, vertex-wise and ROI-average-wise.

For each subject x hemisphere, the 7 condition betas from Sara's FS-FAST
fsaverage GLMs (read-only) are fit as a function of load
x = (ratio - 0.10) / 0.36  in [0, 1]  with four nested-ish models:

  flat      b = a                                          k = 1
  linear    b = a + s*x                                    k = 2
  exp       b = a + s*(1 - exp(-x/tau))/(1 - exp(-1/tau))  k = 3   (compressive;
            tau -> large approaches linear, small tau saturates early)
  NR        b = a + s * x^n / (x^n + c50^n)                k = 4   (Naka-Rushton)

s may be negative (monotonic decrease allowed). Nonlinear parameters (tau;
c50, n) are grid-searched; a, s are solved by least squares for every grid
value at once (profile likelihood), so all ~164k vertices fit in seconds.
Model comparison: AICc = N ln(RSS/N) + 2k + 2k(k+1)/(N-k-1), N = 7.

Also adds subject-specific pRF ROIs (V1-V3 x fovea 0.5-1.5 / periphery 3-7.5
deg, pRF R2 > 10%) from ../prf_fsaverage; S9 has no pRF -> Benson template
eccentricity within Wang V1-V3 (flagged in the output).

Outputs (../outputs):
  roi_betas_long_prf.csv     pRF-ROI betas (same format as roi_betas_long.csv)
  roi_model_fits.csv         per subject x ROI (hemi-averaged) model fits + AICc
  roi_model_group.csv        per ROI: group-mean-curve fits, summed AICc across
                             subjects, n subjects best per model
  vertex_model_summary.csv   per subject x ROI: fraction of vertices best-fit by
                             each model; median dAICc(flat - best non-flat)
  maps/{lh,rh}.*.mgz         fsaverage maps: mean dAICc(flat - exp) across
                             subjects, n subjects with dAICc > 2, mean tau
"""
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_load_betas as base  # noqa: E402  (atlas ROI definitions, paths)

ROOT, PROJ, OUT = base.ROOT, base.PROJ, base.OUT
PRF = f"{ROOT}/prf_fsaverage"
SUBJECTS, HEMIS, RATIOS = base.SUBJECTS, base.HEMIS, base.RATIOS
NV = 163842
X = (RATIOS - RATIOS[0]) / (RATIOS[-1] - RATIOS[0])
N = len(X)
PRF_R2_MIN = 10.0  # pRF R2 is stored in percent
TAUS = np.logspace(-1.5, 1.0, 40)
C50S = np.logspace(-1.5, 0.3, 30)
NS = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MODELS = ["flat", "linear", "exp", "NR"]
K = {"flat": 1, "linear": 2, "exp": 3, "NR": 4}


def aicc(rss, k):
    rss = np.maximum(rss, 1e-12)
    return N * np.log(rss / N) + 2 * k + 2 * k * (k + 1) / (N - k - 1)


def fit_basis(Y, g):
    """Least squares of Y (N x V) on [1, g]; returns rss (V,), a, s."""
    A = np.column_stack([np.ones(N), g])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    rss = ((Y - A @ coef) ** 2).sum(0)
    return rss, coef[0], coef[1]


def fit_all(Y):
    """Fit all models to Y (N x V). Returns dict of per-model arrays."""
    V = Y.shape[1]
    out = {}
    rss0 = ((Y - Y.mean(0)) ** 2).sum(0)
    out["flat"] = dict(rss=rss0)
    rss, a, s = fit_basis(Y, X)
    out["linear"] = dict(rss=rss, slope=s)
    best = dict(rss=np.full(V, np.inf), s=np.zeros(V), tau=np.zeros(V))
    for tau in TAUS:
        g = (1 - np.exp(-X / tau)) / (1 - np.exp(-1 / tau))
        rss, a, s = fit_basis(Y, g)
        m = rss < best["rss"]
        best["rss"][m], best["s"][m], best["tau"][m] = rss[m], s[m], tau
    out["exp"] = best
    best = dict(rss=np.full(V, np.inf), s=np.zeros(V), c50=np.zeros(V), n=np.zeros(V))
    for n in NS:
        xn = X ** n
        for c in C50S:
            g = xn / (xn + c ** n)
            rss, a, s = fit_basis(Y, g)
            m = rss < best["rss"]
            best["rss"][m], best["s"][m], best["c50"][m], best["n"][m] = rss[m], s[m], c, n
    out["NR"] = best
    for mdl in MODELS:
        out[mdl]["aicc"] = aicc(out[mdl]["rss"], K[mdl])
    return out


def best_model(fits):
    A = np.vstack([fits[m]["aicc"] for m in MODELS])
    return A.argmin(0), A


def prf_rois(s, hemi, wang_rois):
    """Subject-specific V1-V3 x eccentricity-band masks on fsaverage."""
    rois, source = {}, "pRF"
    sid = f"S{s:03d}"
    ecc_f = f"{PRF}/{hemi}.{sid}.pRF_ecc.fsaverage.mgz"
    if not os.path.exists(ecc_f):
        source = "template"
        ecc = base.load_mgz(f"{base.ATL}/benson2014/{hemi}.benson14_eccen.v4_0.mgz")
        for area in ("V1", "V2", "V3"):
            for band, (lo, hi) in base.ECC_BANDS.items():
                rois[f"{area}_{band}"] = wang_rois[("wang2015", area)] & (ecc > lo) & (ecc < hi)
        return rois, source
    ecc = base.load_mgz(ecc_f)
    r2 = base.load_mgz(f"{PRF}/{hemi}.{sid}.pRF_R2.fsaverage.mgz")
    for area in ("V1", "V2", "V3"):
        lab = nib.freesurfer.read_label(f"{PRF}/{hemi}.{sid}.pRF.{area.lower()}.fsaverage.label")
        m = np.zeros(NV, bool)
        m[lab] = True
        for band, (lo, hi) in base.ECC_BANDS.items():
            rois[f"{area}_{band}"] = m & (ecc > lo) & (ecc < hi) & (r2 > PRF_R2_MIN)
    return rois, source


def main():
    os.makedirs(f"{OUT}/maps", exist_ok=True)
    atlas = {h: base.build_rois(h) for h in HEMIS}
    prf_rows, vert_rows, roi_curves = [], [], {}
    maps = {h: {"dA_exp": [], "dA_best": [], "tau": [], "sig": []} for h in HEMIS}

    for s in SUBJECTS:
        for h in HEMIS:
            f = f"{PROJ}/sess0{s}/task/attModulated.sm5.{h}/beta.nii.gz"
            beta = np.asarray(nib.load(f).dataobj).reshape(NV, -1)[:, :7].T  # 7 x V
            valid = np.abs(beta).sum(0) > 0
            fits = fit_all(beta[:, valid])
            bi, A = best_model(fits)
            # full-length vertex arrays
            dA_exp = np.full(NV, np.nan); dA_best = np.full(NV, np.nan); tau = np.full(NV, np.nan)
            dA_exp[valid] = fits["flat"]["aicc"] - fits["exp"]["aicc"]
            dA_best[valid] = fits["flat"]["aicc"] - A[1:].min(0)
            tau[valid] = np.where(fits["exp"]["s"] != 0, fits["exp"]["tau"], np.nan)
            maps[h]["dA_exp"].append(dA_exp); maps[h]["dA_best"].append(dA_best)
            maps[h]["tau"].append(tau); maps[h]["sig"].append(np.sign(fits["exp"]["s"]))
            best_full = np.full(NV, -1); best_full[valid] = bi
            sign_full = np.zeros(NV); sign_full[valid] = np.sign(fits["linear"]["slope"])

            rois = {f"{a}|{r}": m for (a, r), m in atlas[h].items()}
            prf, src = prf_rois(s, h, atlas[h])
            rois.update({f"prf|{r}": m for r, m in prf.items()})
            for key, m in rois.items():
                m = m & valid
                n = int(m.sum())
                if n < base.MIN_VERTICES:
                    continue
                at, roi = key.split("|")
                if at == "prf":
                    for r, b in zip(RATIOS, beta[:, m].mean(1)):
                        prf_rows.append((s, h, "prf", roi, n, r, float(b), src))
                roi_curves[(s, h, key)] = beta[:, m].mean(1)
                bm = best_full[m]
                vert_rows.append(dict(
                    subject=s, hemi=h, atlas=at, roi=roi, n_vertices=n,
                    **{f"frac_{mdl}": float((bm == i).mean()) for i, mdl in enumerate(MODELS)},
                    frac_increasing=float((sign_full[m] > 0).mean()),
                    median_dAICc_flat_minus_best=float(np.nanmedian(dA_best[m])),
                    median_dAICc_flat_minus_exp=float(np.nanmedian(dA_exp[m])),
                    frac_dAICc_exp_gt2=float((dA_exp[m] > 2).mean()),
                ))
        print(f"S{s} done", flush=True)

    pd.DataFrame(prf_rows, columns=["subject", "hemi", "atlas", "roi", "n_vertices", "ratio",
                                    "beta", "ecc_source"]).to_csv(f"{OUT}/roi_betas_long_prf.csv", index=False)
    vs = pd.DataFrame(vert_rows)
    vs.to_csv(f"{OUT}/vertex_model_summary.csv", index=False)

    # group vertex maps
    for h in HEMIS:
        for name, arrs in maps[h].items():
            if name == "sig":
                continue
            stack = np.vstack(arrs)
            img = nib.MGHImage(np.nanmean(stack, 0).astype(np.float32).reshape(NV, 1, 1), np.eye(4))
            nib.save(img, f"{OUT}/maps/{h}.mean_{name}.mgz")
        n2 = (np.vstack(maps[h]["dA_exp"]) > 2).sum(0).astype(np.float32)
        nib.save(nib.MGHImage(n2.reshape(NV, 1, 1), np.eye(4)), f"{OUT}/maps/{h}.nsubj_dAICc_exp_gt2.mgz")

    # ROI-average fits: hemi-averaged per subject (ROIs present in both hemis)
    keys = sorted({k for (_, _, k) in roi_curves})
    fit_rows = []
    for key in keys:
        for s in SUBJECTS:
            if not all((s, h, key) in roi_curves for h in HEMIS):
                break
        else:
            curves = np.column_stack([np.mean([roi_curves[(s, h, key)] for h in HEMIS], 0) for s in SUBJECTS])
            fits = fit_all(curves)
            gfit = fit_all(curves.mean(1, keepdims=True))
            at, roi = key.split("|")
            for j, s in enumerate(SUBJECTS):
                fit_rows.append(dict(subject=s, atlas=at, roi=roi,
                                     **{f"aicc_{m}": float(fits[m]["aicc"][j]) for m in MODELS},
                                     **{f"rss_{m}": float(fits[m]["rss"][j]) for m in MODELS},
                                     slope_linear=float(fits["linear"]["slope"][j]),
                                     exp_tau=float(fits["exp"]["tau"][j]), exp_s=float(fits["exp"]["s"][j]),
                                     nr_c50=float(fits["NR"]["c50"][j]), nr_n=float(fits["NR"]["n"][j]),
                                     nr_s=float(fits["NR"]["s"][j])))
            fit_rows.append(dict(subject="groupmean", atlas=at, roi=roi,
                                 **{f"aicc_{m}": float(gfit[m]["aicc"][0]) for m in MODELS},
                                 **{f"rss_{m}": float(gfit[m]["rss"][0]) for m in MODELS},
                                 slope_linear=float(gfit["linear"]["slope"][0]),
                                 exp_tau=float(gfit["exp"]["tau"][0]), exp_s=float(gfit["exp"]["s"][0]),
                                 nr_c50=float(gfit["NR"]["c50"][0]), nr_n=float(gfit["NR"]["n"][0]),
                                 nr_s=float(gfit["NR"]["s"][0])))
    fr = pd.DataFrame(fit_rows)
    fr.to_csv(f"{OUT}/roi_model_fits.csv", index=False)

    subj = fr[fr.subject != "groupmean"].copy()
    A = subj[[f"aicc_{m}" for m in MODELS]].to_numpy()
    subj["best"] = np.array(MODELS)[A.argmin(1)]
    grp = []
    for (at, roi), d in subj.groupby(["atlas", "roi"]):
        g = fr[(fr.subject == "groupmean") & (fr.atlas == at) & (fr.roi == roi)].iloc[0]
        row = dict(atlas=at, roi=roi, n=len(d))
        for m in MODELS:
            row[f"sum_aicc_{m}"] = float(d[f"aicc_{m}"].sum())
            row[f"nbest_{m}"] = int((d.best == m).sum())
            row[f"group_aicc_{m}"] = float(g[f"aicc_{m}"])
        sums = np.array([row[f"sum_aicc_{m}"] for m in MODELS])
        gm = np.array([row[f"group_aicc_{m}"] for m in MODELS])
        row["best_summed"] = MODELS[int(sums.argmin())]
        row["dAICc_summed_flat_minus_best"] = float(sums[0] - sums[1:].min())
        row["best_groupmean"] = MODELS[int(gm.argmin())]
        row["dAICc_group_flat_minus_best"] = float(gm[0] - gm[1:].min())
        row["group_exp_tau"] = float(g.exp_tau)
        row["group_nr_c50"] = float(g.nr_c50)
        row["group_nr_n"] = float(g.nr_n)
        grp.append(row)
    pd.DataFrame(grp).to_csv(f"{OUT}/roi_model_group.csv", index=False)
    print("wrote roi_betas_long_prf.csv, vertex_model_summary.csv, roi_model_fits.csv, roi_model_group.csv, maps/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
