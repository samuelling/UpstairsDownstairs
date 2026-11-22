#!/usr/bin/env python3
"""Thalamic nuclei (LGN, MGN, pulvinar, ...) under central load, with ROI-reliability measures.
Cluster batch job (needs `module load freesurfer/8.2.0 fsl/6.0.4 python3/3.10.12`).

ROIs
  FreeSurfer 8 thalamic-nuclei segmentation (segment_subregions thalamus) of each subject's
  anatomy, grouped into LGN, MGN, pulvinar, mediodorsal, lateral posterior, anterior, ventral,
  intralaminar and whole thalamus, per hemisphere. Independent definition for LGN and MGN:
  the Juelich histological probabilistic atlas (FSL), mapped MNI152 -> MNI305 -> subject via
  talairach.xfm.
Sampling
  Each run's motion-corrected, UNSMOOTHED native 2 mm data (fmcpr.nii.gz). The session
  registration (register.dof6.lta, func voxel -> anatomical voxel) maps 27 sub-samples per
  functional voxel onto the 1 mm segmentation, giving the fraction of each functional voxel
  inside each nucleus (partial volume, PV). ROI time series: mean over voxels with PV >= 0.5
  (main), plus PV >= 0.25, PV-weighted, and (LGN/MGN) Juelich-probability-weighted variants.
GLM per run and ROI
  Intensity-normalized to a global mean of 100; 16-s blocks convolved with the SPM HRF plus
  quadratic drift (the original settings); beta ~ % signal change. Localizer runs (annulus
  on/off) are fit the same way.
Outputs (../thalamus/)
  thal_runs.csv        per run x ROI x variant betas (task and localizer)
  thal_stats.csv       group load effects (fitted 10% -> 46%), with and without run position
  thal_arf.csv         power-law exponent of the load function on the ratio / evidence /
                       behavioral-difficulty axes, and AICc vs flat and linear
  thal_reliability.csv per nucleus: segmentation agreement across algorithm versions (Dice),
                       agreement with the histological atlas (Dice, mean atlas probability,
                       centroid distance), functional voxels and PV, split-half reliability,
                       localizer response, and spread of the load effect across ROI variants
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd, nibabel as nib
from scipy import stats
from scipy.stats import gamma

PROJ = "/projectnb/vision/CentralAttentionManipulation"
ROOT = "/projectnb/vision/CentralAttentionManipulation_reanalysis"
FSS = f"{ROOT}/fs_subjects"
OUT = f"{ROOT}/thalamus"; os.makedirs(OUT, exist_ok=True)
SUBJECTS = [1, 3, 4, 7, 9, 11, 12, 13]
TEST = "--test" in sys.argv
TR = 2.0
LABEL = {"10Perc": .10, "30Perc": .30, "38Perc": .38, "40Perc": .40, "42Perc": .42, "44Perc": .44, "46Perc": .46}
SPAN = 0.36
GROUPS = {"LGN": ["LGN"], "MGN": ["MGN"], "Pulvinar": ["PuA", "PuI", "PuL", "PuM", "PuMm", "PuMl"], "Mediodorsal": ["MDl", "MDm"],
          "Lateral posterior": ["LP"], "Anterior": ["AV", "LD"], "Ventral": ["VA", "VAmc", "VLa", "VLp", "VM", "VPL"],
          "Intralaminar": ["CeM", "CL", "CM", "Pf", "Pc"]}
# FreeSurfer LUT codes for the nuclei
LUT = {}
for line in open(os.path.join(os.environ["FREESURFER_HOME"], "FreeSurferColorLUT.txt")):
    p = line.split()
    if len(p) >= 2 and p[0].isdigit() and 8100 <= int(p[0]) < 8300:
        LUT[p[1]] = int(p[0])
def codes(hemi, names):
    pre = "Left-" if hemi == "lh" else "Right-"
    return [LUT[pre + n] for n in names if pre + n in LUT]
ALLNAMES = sorted({k.split("-", 1)[1] for k in LUT})
GROUPS["Whole thalamus"] = ALLNAMES
# MNI305 -> MNI152 (FreeSurfer CoordinateSystems)
M305_152 = np.array([[0.9975, -0.0073, 0.0176, -0.0429], [0.0146, 1.0009, -0.0024, 1.5496], [-0.0130, -0.0093, 0.9971, 1.1840], [0, 0, 0, 1]])
JUELICH = {("lh", "LGN"): 103, ("rh", "LGN"): 102, ("lh", "MGN"): 106, ("rh", "MGN"): 105}


def read_lta(path):
    txt = open(path).read()
    assert re.search(r"type\s*=\s*0", txt), "expected LINEAR_VOX_TO_VOX"
    rows = txt.split("1 4 4")[1].strip().splitlines()[:4]
    M = np.array([[float(x) for x in r.split()] for r in rows])
    src = re.search(r"src volume info.*?filename\s*=\s*(\S+)", txt, re.S); dst = re.search(r"dst volume info.*?filename\s*=\s*(\S+)", txt, re.S)
    return M, (src.group(1) if src else ""), (dst.group(1) if dst else "")


def read_xfm(path):
    t = open(path).read().split("Linear_Transform =")[1].replace(";", " ").split()
    return np.vstack([np.array(t[:12], float).reshape(3, 4), [0, 0, 0, 1]])


def spm_hrf(dt):
    t = np.arange(0, 32, dt); h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0; return h / h.sum()


def design(par, nt):
    dt = 0.05; hi = np.zeros(int(round(nt * TR / dt))); ratio = None
    for line in open(par):
        p = line.split()
        if len(p) >= 5 and p[1] != "0":
            on, dur = float(p[0]), float(p[2]); ratio = LABEL.get(p[4]); hi[int(on / dt):int((on + dur) / dt)] = 1
    x = np.convolve(hi, spm_hrf(dt))[:len(hi)][(np.arange(nt) * TR / dt).astype(int)]; x /= x.max()
    tt = np.linspace(-1, 1, nt)
    return np.c_[x, np.ones(nt), tt, 1.5 * tt ** 2 - 0.5], ratio


# ---------------- per subject: ROI definitions in functional space ----------------
OFFS = np.array([(a, b, c) for a in (-1 / 3, 0, 1 / 3) for b in (-1 / 3, 0, 1 / 3) for c in (-1 / 3, 0, 1 / 3)])


def seg_file(sid, version="new"):
    if version == "new":
        f = sorted(glob.glob(f"{FSS}/{sid}/mri/ThalamicNuclei*FSvoxelSpace.mgz"))
        if not f and TEST:   # pipeline test before the new segmentation exists
            f = sorted(glob.glob(f"/project/vision/subjects/{sid}/mri/ThalamicNuclei.v10.T1.FSvoxelSpace.mgz"))
    else:
        f = sorted(glob.glob(f"/project/vision/subjects/{sid}/mri/ThalamicNuclei.v10.T1.FSvoxelSpace.mgz"))
    return f[-1] if f else None


def juelich_cache():
    cache = f"{OUT}/juelich_lgn_mgn.npz"
    if not os.path.exists(cache):
        img = nib.load(os.path.join(os.environ["FSLDIR"], "data/atlases/Juelich/Juelich-prob-1mm.nii.gz"))
        vols = {f"{h}_{g}": np.asarray(img.dataobj[..., i]).astype(np.uint8) for (h, g), i in JUELICH.items()}
        np.savez_compressed(cache, affine=img.affine, **vols)
    z = np.load(cache); return z["affine"], {k: z[k] for k in z.files if k != "affine"}


def juelich_subject(sid):
    """Juelich LGN/MGN probabilities warped nonlinearly (FNIRT) into the subject's orig.mgz grid, 0-1; None if missing."""
    d = f"{OUT}/reg/{sid}"; out = {}
    for h in ["lh", "rh"]:
        for g in ["LGN", "MGN"]:
            f = f"{d}/juelich_{h}_{g}.nii.gz"
            if not os.path.exists(f): return None
            out[(h, g)] = np.asarray(nib.load(f).dataobj, dtype=np.float32) / 100.0
    return out


def roi_defs(s):
    sid = open(f"{PROJ}/sess0{s}/subjectname").read().strip()
    segf = seg_file(sid); seg_img = nib.load(segf); seg = np.asarray(seg_img.dataobj).astype(int)
    orig = nib.load(f"{FSS}/{sid}/mri/orig.mgz"); anat_vox2ras = orig.affine
    tal = read_xfm(f"{FSS}/{sid}/mri/transforms/talairach.xfm")
    jaff, jvol = juelich_cache(); jinv = np.linalg.inv(jaff)
    out = {}
    for fsd in ["task", "localizer"]:
        M, src, dst = read_lta(f"{PROJ}/sess0{s}/{fsd}/register.dof6.lta")
        tmpl = nib.load(f"{PROJ}/sess0{s}/{fsd}/template.nii.gz"); fshape = tmpl.shape[:3]
        Minv = np.linalg.inv(M)
        # bounding box of the thalamus in functional voxels
        th = np.argwhere(seg >= 8100)
        fc = (Minv @ np.c_[th, np.ones(len(th))].T).T[:, :3]
        lo = np.maximum(np.floor(fc.min(0)) - 2, 0).astype(int); hi = np.minimum(np.ceil(fc.max(0)) + 2, np.array(fshape) - 1).astype(int)
        grid = np.array(np.meshgrid(*[np.arange(l, h + 1) for l, h in zip(lo, hi)], indexing="ij")).reshape(3, -1).T
        samp = (grid[:, None, :] + OFFS[None]).reshape(-1, 3)
        av = (M @ np.c_[samp, np.ones(len(samp))].T).T[:, :3]
        ai = np.rint(av).astype(int); ok = np.all((ai >= 0) & (ai < np.array(seg.shape)), 1)
        lab = np.zeros(len(ai), int); lab[ok] = seg[ai[ok, 0], ai[ok, 1], ai[ok, 2]]
        lab = lab.reshape(len(grid), 27)
        # histological atlas probability at each sub-sample (anat vox -> scanner RAS -> MNI305 -> MNI152 -> Juelich vox)
        ras = (anat_vox2ras @ np.c_[av, np.ones(len(av))].T)
        mni = (M305_152 @ tal @ ras); jv = np.rint((jinv @ mni).T[:, :3]).astype(int)
        jok = np.all((jv >= 0) & (jv < np.array(next(iter(jvol.values())).shape)), 1)
        jnl = juelich_subject(sid)
        flat = np.ravel_multi_index(grid.T, fshape)
        for hemi in ["lh", "rh"]:
            for g, names in GROUPS.items():
                pv = np.isin(lab, codes(hemi, names)).mean(1)
                k = pv > 0
                out[(fsd, hemi, g, "fs")] = (flat[k], pv[k])
            for g in ["LGN", "MGN"]:
                if jnl is not None:   # nonlinear (preferred): sample the subject-space map at the anatomical sub-samples
                    pr = np.zeros(len(av)); pr[ok] = jnl[(hemi, g)][ai[ok, 0], ai[ok, 1], ai[ok, 2]]
                else:                 # linear fallback through talairach.xfm
                    pr = np.zeros(len(av)); vol = jvol[f"{hemi}_{g}"]; pr[jok] = vol[jv[jok, 0], jv[jok, 1], jv[jok, 2]] / 100.0
                pr = pr.reshape(len(grid), 27).mean(1); k = pr > 0
                out[(fsd, hemi, g, "juelich")] = (flat[k], pr[k])
    return sid, out


# ---------------- segmentation reliability (anatomical space) ----------------
def seg_reliability(s, sid):
    rows = []
    new = nib.load(seg_file(sid)); N = np.asarray(new.dataobj).astype(int)
    oldf = seg_file(sid, "old"); O = np.asarray(nib.load(oldf).dataobj).astype(int) if oldf else None
    orig = nib.load(f"{FSS}/{sid}/mri/orig.mgz"); tal = read_xfm(f"{FSS}/{sid}/mri/transforms/talairach.xfm")
    jaff, jvol = juelich_cache(); jinv = np.linalg.inv(jaff)
    for hemi in ["lh", "rh"]:
        for g, names in GROUPS.items():
            a = np.isin(N, codes(hemi, names)); r = dict(subject=s, hemi=hemi, nucleus=g, vol_mm3=int(a.sum()))
            if O is not None:
                b = np.isin(O, codes(hemi, names)); r["dice_version"] = 2 * (a & b).sum() / max(1, a.sum() + b.sum())
            if g in ("LGN", "MGN"):
                # Juelich probability sampled at every voxel of a box around the FS nucleus
                idx = np.argwhere(a); lo = np.maximum(idx.min(0) - 12, 0); hi = np.minimum(idx.max(0) + 12, np.array(N.shape) - 1)
                box = np.array(np.meshgrid(*[np.arange(l, h + 1) for l, h in zip(lo, hi)], indexing="ij")).reshape(3, -1).T
                mni = M305_152 @ tal @ (orig.affine @ np.c_[box, np.ones(len(box))].T); jv = np.rint((jinv @ mni).T[:, :3]).astype(int)
                vol = jvol[f"{hemi}_{g}"]; ok = np.all((jv >= 0) & (jv < np.array(vol.shape)), 1)
                pr = np.zeros(len(box)); pr[ok] = vol[jv[ok, 0], jv[ok, 1], jv[ok, 2]] / 100.0
                inA = a[box[:, 0], box[:, 1], box[:, 2]]
                jnl = juelich_subject(sid)
                for tag, prob in [("lin", pr)] + ([("nl", jnl[(hemi, g)][box[:, 0], box[:, 1], box[:, 2]])] if jnl is not None else []):
                    atl = prob >= 0.25
                    r[f"dice_atlas25_{tag}"] = 2 * (inA & atl).sum() / max(1, inA.sum() + atl.sum())
                    r[f"mean_atlas_prob_in_fs_{tag}"] = float(prob[inA].mean())
                    if atl.any():
                        c1 = orig.affine[:3, :3] @ box[inA].mean(0); w = prob * atl
                        c2 = orig.affine[:3, :3] @ (box * w[:, None]).sum(0) / w.sum()
                        r[f"centroid_dist_mm_{tag}"] = float(np.linalg.norm(c1 - c2))
            rows.append(r)
    return rows


# ---------------- time series & GLM ----------------
VARIANTS = {"pv50": lambda pv: (pv >= 0.5).astype(float), "pv25": lambda pv: (pv >= 0.25).astype(float), "pvw": lambda pv: pv}


def run_betas(s, defs):
    rows = []
    for fsd in ["task", "localizer"]:
        rdirs = sorted(glob.glob(f"{PROJ}/sess0{s}/{fsd}/0[0-9][0-9]"))
        if TEST: rdirs = rdirs[:2]
        par = "CAM.par" if fsd == "task" else "CAM_localizer.par"
        for rdir in rdirs:
            rn = int(os.path.basename(rdir))
            img = nib.load(f"{rdir}/fmcpr.nii.gz"); Y = np.asarray(img.dataobj, dtype=np.float32); nt = Y.shape[3]
            Y = Y.reshape(-1, nt); vm = Y.mean(1); gm = vm[vm > 0.1 * vm.max()].mean(); Y *= 100.0 / gm
            pf = [f for f in os.listdir(rdir) if f.lower() == par.lower()]
            X, ratio = design(f"{rdir}/{pf[0]}", nt); pinv = np.linalg.pinv(X)
            for (f2, hemi, g, src), (flat, w0) in defs.items():
                if f2 != fsd: continue
                vars_ = VARIANTS if src == "fs" else {"juelich": lambda pv: pv * (pv >= 0.25)}
                for vname, fw in vars_.items():
                    w = fw(w0)
                    if w.sum() < 1e-6: continue
                    y = (Y[flat] * w[:, None]).sum(0) / w.sum()
                    rows.append(dict(subject=s, fsd=fsd, run=rn, ratio=ratio, hemi=hemi, nucleus=g, variant=vname,
                                     beta=float((pinv @ y)[0]), nvox=int((w > 0).sum()), eff_vox=float(w.sum()), mean_pv=float(w0[w > 0].mean())))
            print(f"S{s} {fsd} run {rn}", flush=True)
    return rows


def main():
    seg_rows, all_rows = [], []
    for s in SUBJECTS:
        sid = open(f"{PROJ}/sess0{s}/subjectname").read().strip()
        if not seg_file(sid):
            print("no new segmentation for", sid); continue
        seg_rows += seg_reliability(s, sid)
        _, defs = roi_defs(s)
        all_rows += run_betas(s, defs)
        if TEST: break
    R = pd.DataFrame(all_rows); R.to_csv(f"{OUT}/thal_runs.csv", index=False)
    SR = pd.DataFrame(seg_rows); SR.to_csv(f"{OUT}/thal_seg_reliability.csv", index=False)
    summarize(R, SR)


def summarize(R, SR):
    T = R[R.fsd == "task"].copy(); L = R[R.fsd == "localizer"].copy()
    # hemispheres averaged (main), per subject x run
    Tb = T.groupby(["subject", "run", "ratio", "nucleus", "variant"], as_index=False).beta.mean()
    stats_rows, split_rows, arf_rows = [], [], []
    beh = None
    try:
        import json as _j
        D = _j.loads((s := open(f"{ROOT}/arf/dashboard/data.js").read())[s.index("{"):s.rindex("}") + 1])
        sc = np.array([D["behavior"]["score"]["subjects"][str(x)] for x in D["subjects"]]); dem = 1 - np.maximum.accumulate(sc.mean(0)[::-1])[::-1] / 100
        RAT = np.array(D["ratios"])
        AX = {"ratio": (RAT - RAT[0]) / (RAT[-1] - RAT[0]), "evidence": np.log(0.4 / (0.5 - RAT)) / np.log(0.4 / 0.04), "behavior": (dem - dem[0]) / (dem[-1] - dem[0])}
    except Exception as e:
        print("no behavior axes:", e); AX = None
    QS = np.exp(np.linspace(np.log(0.1), np.log(10), 81))
    def aicc(rss, k, n=7): return n * np.log(max(rss, 1e-12) / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
    for (g, v), d in Tb.groupby(["nucleus", "variant"]):
        sl, slc, curves, odd, even = [], [], [], [], []
        for s, x in d.groupby("subject"):
            x = x.sort_values("run")
            sl.append(np.polyfit(x.ratio, x.beta, 1)[0] * SPAN)
            pos = (x.run - x.run.mean()) / max(1, x.run.std())
            slc.append(np.linalg.lstsq(np.c_[np.ones(len(x)), x.ratio, pos], x.beta.values, rcond=None)[0][1] * SPAN)
            cur = x.groupby("ratio").beta.mean()
            if len(cur) == 7: curves.append(cur.values)
            o = x[x.run % 2 == 1]; e = x[x.run % 2 == 0]
            if o.ratio.nunique() > 2 and e.ratio.nunique() > 2:
                odd.append(np.polyfit(o.ratio, o.beta, 1)[0] * SPAN); even.append(np.polyfit(e.ratio, e.beta, 1)[0] * SPAN)
        for name, vals in [("load", sl), ("load_ctrl_run", slc)]:
            vals = np.array(vals); t, p = stats.ttest_1samp(vals, 0)
            stats_rows.append(dict(nucleus=g, variant=v, measure=name, n=len(vals), mean=vals.mean(), sem=vals.std(ddof=1) / np.sqrt(len(vals)), t=t, p=p, n_pos=int((vals > 0).sum())))
        if len(odd) >= 4:
            r = stats.pearsonr(odd, even).statistic; split_rows.append(dict(nucleus=g, variant=v, split_half_r=r, spearman_brown=2 * r / (1 + r) if r > -1 else np.nan, n=len(odd)))
        if AX is not None and curves:
            C = np.array(curves)
            for ax, xv in AX.items():
                qs, best_lin, best_pow, flat = [], [], [], []
                for y in C:
                    rf = ((y - y.mean()) ** 2).sum(); X1 = np.c_[np.ones(7), xv]; rl = ((y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]) ** 2).sum()
                    bp = min(((((y - (Xq := np.c_[np.ones(7), xv ** q]) @ np.linalg.lstsq(Xq, y, rcond=None)[0]) ** 2).sum()), q) for q in QS)
                    qs.append(bp[1]); flat.append(aicc(rf, 1)); best_lin.append(aicc(rl, 2)); best_pow.append(aicc(bp[0], 3))
                lq = np.log(qs); t, p = stats.ttest_1samp(lq, 0)
                arf_rows.append(dict(nucleus=g, variant=v, axis=ax, q_geomean=float(np.exp(lq.mean())), p_logq=p, n_compressive=int((np.array(qs) < 1).sum()), n=len(qs),
                                     sum_dAICc_flat_minus_best=float(np.sum(np.array(flat) - np.minimum(best_lin, best_pow))),
                                     sum_dAICc_linear_minus_power=float(np.sum(np.array(best_lin) - np.array(best_pow)))))
    ST = pd.DataFrame(stats_rows); ST.to_csv(f"{OUT}/thal_stats.csv", index=False)
    pd.DataFrame(arf_rows).to_csv(f"{OUT}/thal_arf.csv", index=False)
    SP = pd.DataFrame(split_rows)
    # localizer response (annulus on vs fixation), hemispheres averaged, per subject -> group t
    Lb = L.groupby(["subject", "nucleus", "variant"], as_index=False).beta.mean()
    loc = []
    for (g, v), d in Lb.groupby(["nucleus", "variant"]):
        t, p = stats.ttest_1samp(d.beta, 0) if len(d) > 2 else (np.nan, np.nan)
        loc.append(dict(nucleus=g, variant=v, loc_beta=d.beta.mean(), loc_t=t, loc_p=p, loc_n_pos=int((d.beta > 0).sum())))
    LOC = pd.DataFrame(loc)
    vox = T.groupby(["nucleus", "variant"]).agg(func_vox=("nvox", "mean"), eff_vox=("eff_vox", "mean"), mean_pv=("mean_pv", "mean")).reset_index()
    aggs = {c: (c, "mean") for c in SR.columns if c.startswith(("vol_mm3", "dice_", "mean_atlas", "centroid"))}
    seg = SR.groupby("nucleus").agg(**aggs).reset_index()
    main = ST[(ST.measure == "load")].pivot_table(index="nucleus", columns="variant", values="mean")
    spread = (main.max(1) - main.min(1)).rename("load_effect_spread_across_variants").reset_index()
    REL = seg.merge(vox[vox.variant == "pv50"].drop(columns="variant"), on="nucleus", how="left") \
             .merge(SP[SP.variant == "pv50"].drop(columns=["variant", "n"]) if len(SP) else pd.DataFrame({"nucleus": []}), on="nucleus", how="left") \
             .merge(LOC[LOC.variant == "pv50"].drop(columns="variant"), on="nucleus", how="left").merge(spread, on="nucleus", how="left")
    REL.to_csv(f"{OUT}/thal_reliability.csv", index=False)
    pd.set_option("display.width", 250)
    print(ST[ST.variant.isin(["pv50", "juelich"])].round(3).to_string(index=False))
    print(REL.round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(pd.read_csv(f"{OUT}/thal_runs.csv"), pd.read_csv(f"{OUT}/thal_seg_reliability.csv"))
    else:
        main()
