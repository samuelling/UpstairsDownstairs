#!/usr/bin/env python3
"""Export the volumetric group load-effect map for the dashboard's slice viewer (cluster;
needs `module load freesurfer/6.0` for the fsaverage MNI305 2 mm template).

The original analysis also fit the GLM in MNI305 volume space (attModulated.sm5.mni305,
2 mm voxels, 5 mm smoothing; frames 0-6 = the 7 load levels). Per subject and voxel the
load slope x 0.36 = fitted change 10% -> 46% (% signal); mean and one-sample t over the 8
subjects. Underlay: fsaverage mri.2mm/T1.mgz; brain mask and aseg from the same template.

Writes ../outputs/brain/vol.bin (+ vol.json): uint8 anatomy, int16 effect (x1000),
int16 t (x100), uint8 brain mask, uint8 aseg label (subcortical structures only).
"""
import json, os, sys
import numpy as np, nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_load_betas as base

FS = os.environ.get("FREESURFER_HOME", "/share/pkg.7/freesurfer/6.0/install")
T = f"{FS}/subjects/fsaverage/mri.2mm"
OUT = f"{base.OUT}/brain"; os.makedirs(OUT, exist_ok=True)
R = base.RATIOS
pinv = np.linalg.pinv(np.c_[np.ones(7), R])

ref = nib.load(f"{base.PROJ}/sess01/task/attModulated.sm5.mni305/beta.nii.gz")
shape = ref.shape[:3]
eff, used = [], []
for s in base.SUBJECTS:
    f = f"{base.PROJ}/sess0{s}/task/attModulated.sm5.mni305/beta.nii.gz"
    if not os.path.exists(f):
        print("no volumetric GLM for subject", s); continue
    used.append(s)
    b = np.asarray(nib.load(f"{base.PROJ}/sess0{s}/task/attModulated.sm5.mni305/beta.nii.gz").dataobj)[..., :7].reshape(-1, 7)
    eff.append((b @ pinv.T)[:, 1] * (R[-1] - R[0]))
E = np.array(eff); m = E.mean(0); t = m / (E.std(0, ddof=1) / np.sqrt(len(E)) + 1e-9)

# ---- cortex: project the surface group map (fsaverage, all 8 subjects) into the 2 mm volume ribbon ----
import subprocess, tempfile
tmp = tempfile.mkdtemp(dir=OUT)
ctx_eff = np.zeros(int(np.prod(shape)), np.float32); ctx_t = np.zeros_like(ctx_eff); ctx_any = np.zeros(int(np.prod(shape)), bool)
for h in base.HEMIS:
    Es = []
    for s in base.SUBJECTS:
        b = np.asarray(nib.load(f"{base.PROJ}/sess0{s}/task/attModulated.sm5.{h}/beta.nii.gz").dataobj).reshape(163842, -1)[:, :7]
        Es.append((b @ pinv.T)[:, 1] * (R[-1] - R[0]))
    Es = np.array(Es); sm = Es.mean(0); st = sm / (Es.std(0, ddof=1) / np.sqrt(len(Es)) + 1e-9)
    for name, vals in [("eff", sm), ("t", st), ("one", np.ones_like(sm))]:
        src = f"{tmp}/{h}.{name}.mgz"; nib.save(nib.MGHImage(vals.astype(np.float32).reshape(-1, 1, 1), np.eye(4)), src)
        dst = f"{tmp}/{h}.{name}.vol.mgz"
        subprocess.run(["mri_surf2vol", "--surfval", src, "--hemi", h, "--identity", "fsaverage", "--template", f"{T}/orig.mgz",
                        "--fill-projfrac", "0", "1", "0.1", "--o", dst], check=True, stdout=subprocess.DEVNULL, env={**os.environ, "SUBJECTS_DIR": f"{FS}/subjects"})
        v = np.asarray(nib.load(dst).dataobj).reshape(-1)
        if name == "eff": ve = v
        elif name == "t": vt = v
        else: vo = v
    k = vo > 0.5
    ctx_eff[k] = ve[k]; ctx_t[k] = vt[k]; ctx_any |= k
print("cortical ribbon voxels", int(ctx_any.sum()))

anat = np.asarray(nib.load(f"{T}/T1.mgz").dataobj).astype(float)
mask = np.asarray(nib.load(f"{T}/brainmask.mgz").dataobj) > 0
aseg = np.asarray(nib.load(f"{T}/aseg.mgz").dataobj).astype(int)
assert anat.shape == shape, (anat.shape, shape)
SUB = {10: "Left thalamus", 49: "Right thalamus", 11: "Left caudate", 50: "Right caudate", 12: "Left putamen", 51: "Right putamen",
       13: "Left pallidum", 52: "Right pallidum", 17: "Left hippocampus", 53: "Right hippocampus", 18: "Left amygdala", 54: "Right amygdala",
       26: "Left accumbens", 58: "Right accumbens", 28: "Left ventral DC", 60: "Right ventral DC", 16: "Brainstem"}
codes = sorted(SUB); lab = np.zeros(shape, np.uint8)
for k, c in enumerate(codes):
    lab[aseg == c] = k + 1

# combine: cortex from the surface map (8 subjects), everything else from the volume GLM (6 subjects)
vol_valid = (np.abs(m) > 0) & ~ctx_any
src_code = np.zeros(int(np.prod(shape)), np.uint8); src_code[vol_valid] = 2; src_code[ctx_any] = 1
m = np.where(ctx_any, ctx_eff, m); t = np.where(ctx_any, ctx_t, t)
a8 = np.clip(anat / np.percentile(anat[mask], 99.5) * 255, 0, 255).astype(np.uint8)
m16 = np.clip(np.round(m * 1000), -32767, 32767).astype(np.int16).reshape(shape)
t16 = np.clip(np.round(t * 100), -32767, 32767).astype(np.int16).reshape(shape)
parts = [("anat", a8), ("effect", m16), ("t", t16), ("mask", mask.astype(np.uint8)), ("aseg", lab), ("source", src_code.reshape(shape))]
off, info = 0, {}
with open(f"{OUT}/vol.bin", "wb") as f:
    for name, arr in parts:
        buf = np.ascontiguousarray(arr).tobytes(order="C"); info[name] = {"offset": off, "dtype": str(arr.dtype), "scale": {"effect": 1e-3, "t": 1e-2}.get(name, 1)}
        f.write(buf); off += len(buf); pad = (-off) % 4
        if pad: f.write(b"\0" * pad); off += pad
vox2ras = ref.affine.tolist()
json.dump({"shape": list(shape), "order": "C (i, j, k) as stored in the NIfTI, k fastest", "vox2ras": vox2ras, "parts": info, "bytes": off,
           "aseg": ["none"] + [SUB[c] for c in codes], "subjects": used, "surface_subjects": base.SUBJECTS,
           "source": ["none", "cortex (surface map projected to the ribbon, 8 subjects)", "subcortex/cerebellum (volume GLM)"],
           "units": "fitted change 10% -> 46% load (% signal); t(7)"}, open(f"{OUT}/vol.json", "w"), indent=1)
import shutil; shutil.rmtree(tmp, ignore_errors=True)
print("shape", shape, "bytes", off, "effect pct in brain", np.percentile(m.reshape(shape)[mask], [1, 50, 99]).round(2))
