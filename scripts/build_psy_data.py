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
with open(OUT, "w") as f:
    f.write("window.PSY_DATA = " + json.dumps(D, separators=(",", ":")) + ";\n")
print("wrote", OUT, os.path.getsize(OUT), "bytes;", len(D["subjects"]), "subjects,", D["nRuns"], "runs,", D["nTrials"], "trials")
