#!/usr/bin/env python3
"""Build dashboard/data.js from the CSVs in data/ (stdlib only).

Inputs (copied from the SCC reanalysis outputs):
  data/roi_betas_long.csv   subject, hemi, atlas, roi, n_vertices, ratio, beta
  data/roi_slopes.csv       per subject x roi slope fits (hemis averaged)
  data/group_slopes.csv     per roi group stats (all8 / excl12_13)
  data/behavior_by_run.txt  per-run CAM.par vs behavior check: raw score
                            (nansum/len), response rate, accuracy of answered
  data/roi_betas_long_prf.csv, prf_group_slopes.csv   subject-pRF ROIs
  data/roi_model_group.csv, vertex_model_summary.csv, model_recovery.csv
"""
import csv
import json
import math
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "dashboard", "data.js")

SUBJECTS = [1, 3, 4, 7, 9, 11, 12, 13]
SETS = {"all8": SUBJECTS}
KEY_FLIPPED = (12, 13)  # pressed reversed keys in every run (raw accuracy 0-46%)
RATIOS = [0.10, 0.30, 0.38, 0.40, 0.42, 0.44, 0.46]
T975 = {7: 2.3646, 5: 2.5706}  # two-sided 95% t critical values, df = n-1


def mean(xs):
    return sum(xs) / len(xs)


def sem(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))


def rkey(r):
    return f"{float(r):.2f}"


def behavior():
    """Per run: score = correct / all majority changes (misses count wrong),
    response rate, accuracy of answered changes. For key-reversed subjects the
    correct score is resp_rate * (1 - raw accuracy of answered) - i.e. Sara's
    nansum(1 - x)/len. (An earlier version used 100 - raw score, which wrongly
    counted misses as correct.)"""
    per = {k: defaultdict(list) for k in ("score", "resp", "acc")}
    n_runs = 0
    with open(os.path.join(DATA, "behavior_by_run.txt")) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 9 and p[-1] == "ok":
                s, ratio = int(p[0]), rkey(p[4])
                raw, resp, acc = float(p[5]), float(p[6]), float(p[7])
                if s in KEY_FLIPPED:
                    acc = 100 - acc if acc == acc else acc
                    raw = resp * acc / 100 if acc == acc else 0.0
                per["score"][(s, ratio)].append(raw)
                per["resp"][(s, ratio)].append(resp)
                if acc == acc:
                    per["acc"][(s, ratio)].append(acc)
                n_runs += 1
    out = {"nRuns": n_runs, "keyFlipped": list(KEY_FLIPPED)}
    for k in ("score", "resp", "acc"):
        subj = {s: [round(mean(per[k][(s, rkey(r))]), 1) for r in RATIOS] for s in SUBJECTS}
        out[k] = {"subjects": subj, "group": [{"m": round(mean([subj[s][i] for s in SUBJECTS]), 2),
                                                "se": round(sem([subj[s][i] for s in SUBJECTS]), 2)}
                                               for i in range(len(RATIOS))]}
    return out


def curves():
    cell = defaultdict(lambda: defaultdict(list))  # (atlas,roi) -> (subj,ratio) -> [hemi betas]
    nvert = defaultdict(dict)
    for fname in ("roi_betas_long.csv", "roi_betas_long_prf.csv"):
        with open(os.path.join(DATA, fname)) as fh:
            for row in csv.DictReader(fh):
                key = (row["atlas"], row["roi"])
                cell[key][(int(row["subject"]), rkey(row["ratio"]))].append(float(row["beta"]))
                nvert[key][row["hemi"]] = int(row["n_vertices"])  # last subject's count (display only)
    out = {}
    for (atlas, roi), d in cell.items():
        # keep only ROIs present in both hemispheres for every subject (as in the slope fits)
        if any(len(d[(s, rkey(r))]) != 2 for s in SUBJECTS for r in RATIOS):
            continue
        subj = {s: [round(mean(d[(s, rkey(r))]), 4) for r in RATIOS] for s in SUBJECTS}
        grp = {}
        for k, subs in SETS.items():
            grp[k] = [{"m": round(mean([subj[s][i] for s in subs]), 4),
                       "se": round(sem([subj[s][i] for s in subs]), 4)} for i in range(len(RATIOS))]
        out[f"{atlas}|{roi}"] = {"atlas": atlas, "roi": roi, "nVert": nvert[(atlas, roi)],
                                 "subjects": subj, "group": grp}
    return out


def slopes():
    per = defaultdict(dict)
    with open(os.path.join(DATA, "roi_slopes.csv")) as fh:
        for row in csv.DictReader(fh):
            per[f"{row['atlas']}|{row['roi']}"][int(row["subject"])] = round(float(row["slope_per_unit_ratio"]), 4)
    rows = []
    src = []
    for fname in ("group_slopes.csv", "prf_group_slopes.csv"):
        with open(os.path.join(DATA, fname)) as fh:
            src += [r for r in csv.DictReader(fh) if r["subjects"] in SETS]
    for row in src:
            n = int(row["n"])
            m, se = float(row["mean_slope"]), float(row["sem"])
            key = f"{row['atlas']}|{row['roi']}"
            subs = SETS[row["subjects"]]
            rows.append({
                "set": row["subjects"], "atlas": row["atlas"], "roi": row["roi"], "n": n,
                "slope": round(m, 4), "se": round(se, 4),
                "lo": round(m - T975[n - 1] * se, 4), "hi": round(m + T975[n - 1] * se, 4),
                "t": round(float(row["t"]), 3), "p": float(f"{float(row['p']):.3g}"),
                "q": float(f"{float(row['q_fdr_within_atlas']):.3g}"),
                "rho": round(float(row["mean_rho"]), 3), "beta": round(float(row["mean_beta"]), 4),
                "subj": {s: per[key][s] for s in subs if s in per.get(key, {})},
            })
    return rows


def models():
    vert = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(DATA, "vertex_model_summary.csv")) as fh:
        for r in csv.DictReader(fh):
            for k in ("frac_flat", "frac_linear", "frac_exp", "frac_NR", "frac_increasing"):
                vert[(r["atlas"], r["roi"])][k].append(float(r[k]))
    rows = []
    with open(os.path.join(DATA, "roi_model_group.csv")) as fh:
        for r in csv.DictReader(fh):
            v = vert[(r["atlas"], r["roi"])]
            rows.append({
                "atlas": r["atlas"], "roi": r["roi"],
                "nbest": {m: int(r[f"nbest_{m}"]) for m in ("flat", "linear", "exp", "NR")},
                "bestSummed": r["best_summed"], "dSummed": round(float(r["dAICc_summed_flat_minus_best"]), 2),
                "bestGroup": r["best_groupmean"], "dGroup": round(float(r["dAICc_group_flat_minus_best"]), 2),
                "tau": round(float(r["group_exp_tau"]), 3),
                "vert": {k.replace("frac_", ""): round(mean(x), 4) for k, x in v.items()},
            })
    rec = []
    with open(os.path.join(DATA, "model_recovery.csv")) as fh:
        for r in csv.DictReader(fh):
            rec.append({"noise": float(r["noise_sd"]), "truth": r["truth"],
                        **{m: float(r[f"pct_{m}"]) for m in ("flat", "linear", "exp", "NR")}})
    return {"rois": rows, "recovery": rec}


def hemispheres():
    """Per-hemisphere group curves, slope stats (lh, rh, rh-lh), per-hemi vertex model fractions,
    and R-vs-L composites (from hemi_stats.py)."""
    curves = defaultdict(lambda: {"lh": [None] * len(RATIOS), "rh": [None] * len(RATIOS)})
    with open(os.path.join(DATA, "hemi_curves.csv")) as fh:
        for r in csv.DictReader(fh):
            i = [rkey(x) for x in RATIOS].index(rkey(r["ratio"]))
            curves[f"{r['atlas']}|{r['roi']}"][r["hemi"]][i] = {"m": round(float(r["m"]), 4), "se": round(float(r["se"]), 4)}
    vert = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(DATA, "vertex_model_summary.csv")) as fh:
        for r in csv.DictReader(fh):
            for k in ("frac_flat", "frac_linear", "frac_exp", "frac_NR", "frac_increasing"):
                vert[(r["atlas"], r["roi"], r["hemi"])][k].append(float(r[k]))
    slopes = []
    with open(os.path.join(DATA, "hemi_slopes.csv")) as fh:
        for r in csv.DictReader(fh):
            n, m, se = int(r["n"]), float(r["mean_slope"]), float(r["sem"])
            row = {"atlas": r["atlas"], "roi": r["roi"], "hemi": r["hemi"], "n": n,
                   "slope": round(m, 4), "se": round(se, 4),
                   "lo": round(m - T975[n - 1] * se, 4), "hi": round(m + T975[n - 1] * se, 4),
                   "t": round(float(r["t"]), 3), "p": float(f"{float(r['p']):.3g}"), "q": float(f"{float(r['q']):.3g}"),
                   "subj": {s: round(float(r[f"s{s}"]), 4) for s in SUBJECTS}}
            if r["hemi"] in ("lh", "rh"):
                v = vert[(r["atlas"], r["roi"], r["hemi"])]
                row.update(best=r["best_summed"], dSummed=round(float(r["dAICc_summed"]), 2),
                           nLoad=int(float(r["nsubj_load"])), beta=round(float(r["mean_beta"]), 4),
                           vert={k.replace("frac_", ""): round(mean(x), 4) for k, x in v.items()} if v else None)
            slopes.append(row)
    comps = []
    with open(os.path.join(DATA, "hemi_composites.csv")) as fh:
        for r in csv.DictReader(fh):
            comps.append({k: (r[k] if k in ("label", "atlas", "rois") else round(float(r[k]), 4)) for k in r})
    return {"curves": dict(curves), "slopes": slopes, "composites": comps}


def main():
    data = {"ratios": RATIOS, "subjects": SUBJECTS, "sets": SETS,
            "behavior": behavior(), "curves": curves(), "slopes": slopes(), "models": models(),
            "hemi": hemispheres()}
    # per-subject OLS slopes (per unit ratio) for ROIs absent from roi_slopes.csv (pRF ROIs)
    mx = mean(RATIOS)
    for row in data["slopes"]:
        c = data["curves"].get(f"{row['atlas']}|{row['roi']}")
        if not row["subj"] and c:
            row["subj"] = {s: round(sum((x - mx) * (y - mean(c["subjects"][s])) for x, y in zip(RATIOS, c["subjects"][s]))
                                    / sum((x - mx) ** 2 for x in RATIOS), 4) for s in data["sets"][row["set"]]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("// Generated by scripts/build_data.py - do not edit by hand.\n")
        fh.write("window.CAM_DATA = ")
        json.dump(data, fh, separators=(",", ":"))
        fh.write(";\n")
    print(f"wrote {OUT}: {len(data['curves'])} ROI curves, {len(data['slopes'])} slope rows, "
          f"{data['behavior']['nRuns']} behavioral runs, {len(data['models']['rois'])} model rows")


if __name__ == "__main__":
    main()
