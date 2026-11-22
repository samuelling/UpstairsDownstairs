#!/usr/bin/env python3
"""Build dashboard/extra_data.js (run-position control + demand-axis ARF fits) from
data/glm_runpos/ and data/arf_*.csv (stdlib only).

Refresh from the cluster first:
  rsync -a scc:/projectnb/vision/CentralAttentionManipulation_reanalysis/glm_runpos/{group_slopes_runpos.csv,subject_slopes.csv} data/glm_runpos/
  (arf_demand.csv / arf_axes.csv come from scripts/arf_demand.py)
"""
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SPAN = 0.36


def rows(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                try:
                    d[k] = float(v) if v not in ("", "nan") else None
                except ValueError:
                    d[k] = v
            out.append(d)
    return out


G = rows(os.path.join(DATA, "glm_runpos", "group_slopes_runpos.csv"))
keep = {"slopeA", "slopeB", "slopeC", "slopeC_mc", "runB", "runB_mc"}
runpos = [{"atlas": g["atlas"], "roi": g["roi"], "m": g["measure"], "n": int(g["n"]),
           "eff": (g["mean"] * SPAN) if g["measure"].startswith("slope") else g["mean"],
           "se": (g["sem"] * SPAN) if g["measure"].startswith("slope") else g["sem"],
           "t": g["t"], "p": g["p"], "q": g["q"]} for g in G if g["measure"] in keep]
S = rows(os.path.join(DATA, "glm_runpos", "subject_slopes.csv"))
corr = {}
for s in S:
    corr.setdefault(int(s["subject"]), s["corr_ratio_run"])
E = {"runpos": runpos, "corrRatioRun": corr,
     "runTermMc": {"nP05": sum(1 for g in G if g["measure"] == "runB_mc" and g["p"] < .05), "nRois": sum(1 for g in G if g["measure"] == "runB_mc"),
                   "minQ": min(g["q"] for g in G if g["measure"] == "runB_mc"),
                   "nNeg": sum(1 for g in G if g["measure"] == "runB_mc" and float(g["mean"]) < 0)},
     "runTerm": {"nP05": sum(1 for g in G if g["measure"] == "runB" and g["p"] < .05), "minQ": min(g["q"] for g in G if g["measure"] == "runB")},
     "arf": rows(os.path.join(DATA, "arf_demand.csv")), "arfAxes": rows(os.path.join(DATA, "arf_axes.csv"))}
# eccentricity profile + annulus-localizer ROIs (scripts/ecc_localizer.py, cluster)
vm = os.path.join(DATA, "visual_maps")
if os.path.exists(os.path.join(vm, "ecc_profile.csv")):
    ep = rows(os.path.join(vm, "ecc_profile.csv")); lr = rows(os.path.join(vm, "localizer_rois.csv"))
    E["ecc"] = {"group": [r for r in ep if r["kind"] == "group" and r["n_subj"] and r["n_subj"] >= 6],
                "subjects": [{k: r[k] for k in ("subject", "area", "ecc_mid", "effect")} for r in ep if r["kind"] == "subject"]}
    E["loc"] = {"group": [r for r in lr if r["kind"] == "group"],
                "subjects": [{k: r[k] for k in ("subject", "area", "roi", "effect", "median_ecc", "mean_beta")} for r in lr if r["kind"] == "subject"]}
# run-level behavior -> brain and residual coupling (scripts/run_level_coupling.py)
rl = os.path.join(DATA, "run_level")
if os.path.exists(os.path.join(rl, "behavior_brain.csv")):
    bb = rows(os.path.join(rl, "behavior_brain.csv")); cc = rows(os.path.join(rl, "coupling.csv"))
    E["runlevel"] = {"beh": bb, "coup": cc}
out = os.path.join(HERE, "..", "dashboard", "extra_data.js")
with open(out, "w") as f:
    f.write("window.CAM_EXTRA = " + json.dumps(E, separators=(",", ":")) + ";\n")
print("wrote", out, os.path.getsize(out), "bytes")
