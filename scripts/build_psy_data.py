#!/usr/bin/env python3
"""Build dashboard/psy_data.js from data/psychophysics/*.csv (stdlib only).

Refresh the CSVs from the cluster first:
  rsync -a scc:/projectnb/vision/AttentionManipulation_reanalysis/outputs/ data/psychophysics/
"""
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data", "psychophysics")
OUT = os.path.join(HERE, "..", "dashboard", "psy_data.js")


def rows(name):
    with open(os.path.join(SRC, name), newline="") as f:
        out = []
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if v in ("True", "False"):
                    d[k] = v == "True"
                else:
                    try:
                        d[k] = float(v) if v != "" else None
                    except ValueError:
                        d[k] = v
            out.append(d)
        return out


cells = rows("psy_cells.csv")
group = rows("psy_group.csv")
subjects = rows("psy_subjects.csv")
runs = rows("psy_runs.csv")
D = {
    "common": sorted({r["ratio"] for r in group}),
    "group": {s: [r for r in group if r["set"] == s] for s in ("reliable", "all")},
    "cells": [{k: r[k] for k in ("subject", "ratio", "rb_acc", "rb_resp", "od_acc", "quest_thr_median", "pf_thr", "reliable", "common", "n_trials")} for r in cells],
    "subjects": subjects,
    "tests": rows("psy_slopes.csv"),
    "glm": rows("psy_trial_glm.csv"),
    "glmGroup": rows("psy_trial_glm_group.csv"),
    "psychometric": rows("psy_psychometric.csv"),
    "original": rows("psy_original.csv"),
    "nRuns": len({(r["subject"], r["run"]) for r in runs}),
    "nTrials": int(sum(r["n_trials"] for r in cells)),
}
# AttRSF contrast-discrimination pilots (scripts/rsf_extract.py on the cluster, then scripts/rsf_tvc.py)
if os.path.exists(os.path.join(SRC, "rsf_fits.csv")):
    fits = rows("rsf_fits.csv")
    ext = {}
    for r in fits:
        e = ext.setdefault(r["set"], [r["load"], r["load"]]); e[0] = min(e[0], r["load"]); e[1] = max(e[1], r["load"])
    D["rsf"] = {
        "cells": [r for r in rows("rsf_cells.csv")],
        "curves": [r for r in rows("rsf_curves.csv") if r["hyp"] in ("rg", "cg", "none") and r["load"] in ext.get(r["set"], [])],
        "fits": [r for r in fits if r["load"] in ext[r["set"]]],
        "models": sorted({(r["set"], r["hyp"], r["dAICc"], int(r["k"]), int(r["N"])) for r in fits}),
        "extremes": ext,
    }
with open(OUT, "w") as f:
    f.write("window.PSY_DATA = " + json.dumps(D, separators=(",", ":")) + ";\n")
print("wrote", OUT, os.path.getsize(OUT), "bytes;", len(D["subjects"]), "subjects,", D["nRuns"], "runs,", D["nTrials"], "trials")
