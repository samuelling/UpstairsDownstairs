#!/usr/bin/env python3
"""Export inflated fsaverage6 surfaces, ROI labels and a group load-effect map for the
dashboard's brain viewer (cluster; needs `module load freesurfer/6.0`).

fsaverage6 (40,962 vertices/hemisphere) is nested in fsaverage: its vertices are the first
40,962 fsaverage vertices, so every fsaverage label/map is transferred by taking those
entries. ROI definitions are exactly those used in the analyses (roi_load_betas.build_rois).

Writes to ../outputs/brain/:
  {lh,rh}.bin   float32 positions (nV*3), sulc (nV), effect (nV), t (nV);
                uint16 faces (nF*3); uint8 labels, one array of nV per atlas
  meta.json     sizes, byte offsets and label names for each atlas
Group load-effect map: per subject and vertex, the slope of the 7 load betas (original
FS-FAST GLMs) x 0.36 = fitted change 10% -> 46% (% signal); mean and one-sample t over
the 8 subjects.
"""
import json, os, sys
import numpy as np, nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_load_betas as base

FS = os.environ.get("FREESURFER_HOME", "/share/pkg.7/freesurfer/6.0/install")
OUT = f"{base.OUT}/brain"; os.makedirs(OUT, exist_ok=True)
N6 = 40962
R = base.RATIOS


def atlases_for(rois):
    """Group build_rois output into viewer atlases: {atlas: [(name, mask)]} (first match wins)."""
    A = {"wang2015": [], "bands": [], "schaeferNet": [], "glasser2016": []}
    for (atlas, roi), m in rois.items():
        if atlas == "wang2015":
            (A["bands"] if roi.endswith(("_fovea", "_periph")) else A["wang2015"]).append((roi, m))
        elif atlas == "schaefer2018" and "_" not in roi:
            A["schaeferNet"].append((roi, m))
        elif atlas == "glasser2016":
            A["glasser2016"].append((roi, m))
    return A


meta = {"nV": N6, "atlases": {}, "hemis": {}}
# group effect map
eff = {h: [] for h in base.HEMIS}
Xr = np.c_[np.ones(7), R]
pinv = np.linalg.pinv(Xr)
for s in base.SUBJECTS:
    for h in base.HEMIS:
        b = np.asarray(nib.load(f"{base.PROJ}/sess0{s}/task/attModulated.sm5.{h}/beta.nii.gz").dataobj).reshape(163842, -1)[:N6, :7]
        eff[h].append((b @ pinv.T)[:, 1] * (R[-1] - R[0]))

names_by_atlas = {}
for h in base.HEMIS:
    coords, faces = nib.freesurfer.read_geometry(f"{FS}/subjects/fsaverage6/surf/{h}.inflated")
    sulc = nib.freesurfer.read_morph_data(f"{FS}/subjects/fsaverage6/surf/{h}.sulc")
    E = np.array(eff[h]); m = E.mean(0); sd = E.std(0, ddof=1); t = m / (sd / np.sqrt(len(E)) + 1e-9)
    rois = base.build_rois(h)
    A = atlases_for(rois)
    labels = {}
    for atlas, items in A.items():
        names = names_by_atlas.setdefault(atlas, [])
        lab = np.zeros(N6, np.uint8)
        for name, mask in items:
            if name not in names:
                names.append(name)
            k = names.index(name) + 1
            sel = mask[:N6] & (lab == 0)
            lab[sel] = k
        labels[atlas] = lab
    parts = [("positions", coords.astype(np.float32).ravel()), ("sulc", sulc.astype(np.float32)), ("effect", m.astype(np.float32)), ("t", t.astype(np.float32)),
             ("faces", faces.astype(np.uint16).ravel())] + [(f"label:{a}", labels[a]) for a in A]
    off, info = 0, {}
    with open(f"{OUT}/{h}.bin", "wb") as f:
        for name, arr in parts:
            buf = arr.tobytes(); info[name] = {"offset": off, "length": int(arr.size), "dtype": str(arr.dtype)}
            f.write(buf); off += len(buf)
            pad = (-off) % 4
            if pad: f.write(b"\0" * pad); off += pad
    meta["hemis"][h] = {"nF": int(len(faces)), "parts": info, "bytes": off}
    print(h, "vertices", len(coords), "faces", len(faces), "bytes", off, "effect range", np.percentile(m, [1, 99]).round(2))
meta["atlases"] = names_by_atlas
meta["effect"] = {"subjects": len(base.SUBJECTS), "units": "fitted change 10% -> 46% load (% signal)"}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
print({a: len(n) for a, n in names_by_atlas.items()})
