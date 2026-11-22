#!/usr/bin/env python3
"""Extract staircase thresholds from the AttRSF contrast-discrimination pilots (cluster).

Reads the pilot folders under AttentionManipulation/ToBeTrashed (read-only), codes the two
pilot subjects as A and B (key written only to the cluster reanalysis folder), and writes
outputs/rsf_pilot_cells_raw.csv: one row per run x blue-count staircase with the pedestal
contrast, central load (minority share), threshold (QUEST estimate - pedestal) and central
accuracy. Then run scripts/rsf_tvc.py on that file.
"""
import glob, os, re
import numpy as np, pandas as pd, scipy.io as sio

SRC = "/projectnb/vision/AttentionManipulation/ToBeTrashed"
ROOT = "/projectnb/vision/AttentionManipulation_reanalysis"
L = lambda f: sio.loadmat(f, squeeze_me=True, struct_as_record=False)
SETS = {"RedBlue100RectDoubleTask": "100", "RedBlue200RectDoubleTask": "200"}

folders = sorted(d for d in glob.glob(f"{SRC}/DataSbj*") if any(d.endswith(k) for k in SETS))
names = sorted({re.sub(r"^DataSbj|(" + "|".join(SETS) + r")$", "", os.path.basename(d)) for d in folders})
code = {n: chr(ord("A") + i) for i, n in enumerate(names)}
pd.DataFrame({"code": list(code.values()), "name": list(code.keys())}).to_csv(f"{ROOT}/rsf_subject_key_DO_NOT_SHARE.csv", index=False)

rows = []
for d in folders:
    base = os.path.basename(d); suffix = next(k for k in SETS if base.endswith(k))
    subj = code[re.sub(r"^DataSbj|" + suffix + "$", "", base)]
    for f in glob.glob(f"{d}/*.mat"):
        m = L(f); ev = np.atleast_1d(m["events"])[0]; r = np.atleast_1d(m["results"])[0]; n = m["rects"].nTotal
        ped = float(ev.cntr); counts = np.atleast_1d(m["rects"].nGreenVec); est = 10 ** np.atleast_1d(r.finalEstimate)
        ng = np.atleast_1d(ev.numGreens); rg = np.atleast_1d(r.correctRespRG)
        run = int(re.search(r"_Run(\d+)_", f).group(1))
        for c, e in zip(counts, est):
            rows.append(dict(set=f"{subj}-{SETS[suffix]}", run=run, ped=round(ped, 4), ratio=round(min(c, n - c) / n, 3), thr=e - ped, rg_acc=rg[ng == c].mean()))
os.makedirs(f"{ROOT}/outputs", exist_ok=True)
pd.DataFrame(rows).to_csv(f"{ROOT}/outputs/rsf_pilot_cells_raw.csv", index=False)
print(len(rows), "staircases;", sorted({r["set"] for r in rows}))
