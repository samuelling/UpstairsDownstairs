#!/usr/bin/env python3
"""Read-only check: does each fMRI run's CAM.par condition match the ratio
stored in the matching behavioral file (TheData.rects.rectRatios)?

Assumes behavioral Run N  <->  scan run folder N (zero-padded 001..),
which matches the per-subject run counts. Writes nothing; prints a table.
Run on the SCC:  python3 check_par_vs_behavior.py
"""
import glob
import os
import re
import sys

import scipy.io as sio

PROJ = "/projectnb/vision/CentralAttentionManipulation"
SUBJECTS = [1, 3, 4, 7, 9, 11, 12, 13]
LABEL_TO_RATIO = {"10Perc": .10, "30Perc": .30, "38Perc": .38, "40Perc": .40,
                  "42Perc": .42, "44Perc": .44, "46Perc": .46}


def par_ratio(par_path):
    labels = set()
    with open(par_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 5 and parts[1] != "0":
                labels.add(parts[4])
    if len(labels) != 1:
        return None, ",".join(sorted(labels)) or "none"
    lab = labels.pop()
    return LABEL_TO_RATIO.get(lab), lab


def beh_ratio_and_perf(mat_path):
    d = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    td = d["TheData"]
    ratio = float(td.rects.rectRatios)
    # correctResp: one cell per block, one entry per majority-color change
    # (1 correct, 0 wrong, NaN no response). Sara's scoring: nansum/length
    # (misses count as wrong); a run is dropped if nanmean == 0.
    import numpy as np
    vals = np.concatenate([np.atleast_1d(np.asarray(b, dtype=float))
                           for b in np.atleast_1d(td.output.correctResp)])
    # Raw values only - no key-flip here. Downstream, a key-reversed subject's
    # score is nansum(1 - x)/len (misses stay wrong), NOT 100 - raw score.
    perf = float(np.nansum(vals) / len(vals)) if len(vals) else None
    answered = ~np.isnan(vals)
    resp_rate = float(answered.mean()) if len(vals) else None
    acc_answered = float(vals[answered].mean()) if answered.any() else float("nan")
    return ratio, perf, resp_rate, acc_answered


def main():
    n_bad = 0
    print(f"{'subj':>4} {'run':>4} {'par_label':>10} {'par':>5} {'beh':>5} {'pctCorr':>7} {'respPct':>7} {'accAns':>7}  status")
    for s in SUBJECTS:
        runs = sorted(glob.glob(f"{PROJ}/sess0{s}/task/0[0-9][0-9]"))
        for rdir in runs:
            rn = int(os.path.basename(rdir))
            pr, lab = par_ratio(f"{rdir}/CAM.par")
            mat = f"{PROJ}/Behavioral/{s}/Data_CentralAttMod_Run{rn}_{s}.mat"
            if not os.path.exists(mat):
                print(f"{s:>4} {rn:>4} {lab:>10} {'':>5} {'':>5} {'':>7} {'':>7} {'':>7}  NO_BEHAVIOR_FILE")
                n_bad += 1
                continue
            br, perf, rrate, acc = beh_ratio_and_perf(mat)
            ok = pr is not None and abs(pr - br) < 1e-6
            n_bad += not ok
            pstr = f"{perf*100:6.1f}" if perf is not None else "   n/a"
            rstr = f"{rrate*100:6.1f}" if rrate is not None else "   n/a"
            astr = f"{acc*100:6.1f}" if acc == acc else "   nan"
            print(f"{s:>4} {rn:>4} {lab:>10} {pr if pr is not None else 'NA':>5} {br:>5.2f} {pstr:>7} {rstr:>7} {astr:>7}  {'ok' if ok else 'MISMATCH'}")
    print(f"\nmismatches/missing: {n_bad}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
