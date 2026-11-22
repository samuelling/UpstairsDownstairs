#!/usr/bin/env python3
"""Run-level analyses on the run-by-run task betas (glm_runpos/run_betas.csv).

(1) Behavior -> brain. Within each subject, holding load fixed, do runs in which the
    subject detected more of the majority flips show more dorsal-attention activity, or
    more peripheral suppression? Model per subject x ROI:
        beta_run ~ load-level dummies + run position + detection (z within subject)
    Detection (% of flips answered) is used because it is unaffected by the key
    reversal in S12/S13. The detection coefficient (% signal per SD) is t-tested
    across subjects.
(4) IPS <-> periphery. Within each subject, after removing load-level means and run
    position, are runs with more dorsal-attention activity than expected also runs
    with less peripheral V1-V3 activity? Pearson r across runs, Fisher z, t-test across
    subjects. Because all ROIs share run-wide fluctuations (arousal, global signal), the
    same is also computed as a partial correlation controlling a global covariate that
    shares no ROI with either member: the mean residual of the Schaefer default, control
    and salience networks (control and salience only when the pair includes the default
    network).

Input:  data/glm_runpos/run_betas.csv, data/behavior_by_run.txt
Output: data/run_level/behavior_brain.csv, coupling.csv, coupling_pairs.csv
"""
import os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(DATA, "run_level"); os.makedirs(OUT, exist_ok=True)

R = pd.read_csv(os.path.join(DATA, "glm_runpos", "run_betas.csv"))
B = R.groupby(["subject", "run", "ratio", "atlas", "roi"], as_index=False).beta.mean()   # hemispheres averaged
W = B.pivot_table(index=["subject", "run", "ratio"], columns=["atlas", "roi"], values="beta")
W.columns = [f"{a}|{r}" for a, r in W.columns]
W = W.reset_index()

# composites
dors = [f"wang2015|{r}" for r in ["IPS0", "IPS1", "IPS2", "IPS3", "IPS4/5", "SPL1", "FEF"]]
W["Dorsal attention (Wang IPS–FEF)"] = W[dors].mean(axis=1)
W["Foveal V1–V3"] = W[[f"wang2015|{a}_fovea" for a in ["V1", "V2", "V3"]]].mean(axis=1)
W["Peripheral V1–V3"] = W[[f"wang2015|{a}_periph" for a in ["V1", "V2", "V3"]]].mean(axis=1)
glob_cols = [c for c in W.columns if c.startswith("schaefer2018|") and "_" not in c.split("|")[1]
             and c.split("|")[1].startswith(("Default", "Cont", "SalVentAttn"))]
W["Global (default/control/salience)"] = W[glob_cols].mean(axis=1)
# covariate for pairs involving the default network: control + salience only (no overlap with either member)
W["Global (control/salience)"] = W[[c for c in glob_cols if not c.split("|")[1].startswith("Default")]].mean(axis=1)

# behavior per run
beh = pd.read_fwf(os.path.join(DATA, "behavior_by_run.txt"), comment="m", skipfooter=0)
beh = beh[pd.to_numeric(beh["subj"], errors="coerce").notna()].copy()
beh = beh.rename(columns={"subj": "subject", "respPct": "detect"})
for c in ["subject", "run", "detect"]:
    beh[c] = pd.to_numeric(beh[c], errors="coerce")
W = W.merge(beh[["subject", "run", "detect"]], on=["subject", "run"], how="left")

ROIS = ["Dorsal attention (Wang IPS–FEF)", "wang2015|IPS0", "wang2015|IPS2", "wang2015|FEF", "Foveal V1–V3",
        "Peripheral V1–V3", "wang2015|V1_periph", "wang2015|V2_periph", "schaefer2018|DefaultA",
        "schaefer2018|DorsAttnA", "Global (default/control/salience)"]


def design(d):
    L = pd.get_dummies(d.ratio, prefix="L", drop_first=False).astype(float).values
    pos = (d.run - d.run.mean()) / d.run.std()
    return L, pos.values


# ---------- (1) behavior -> brain ----------
rows = []
for s, d in W.groupby("subject"):
    d = d.dropna(subset=["detect"]).sort_values("run")
    if d.detect.std() == 0 or len(d) < 10:
        continue
    L, pos = design(d)
    # detection centred within load level, then scaled by its within-subject SD
    dz = d.detect - d.groupby("ratio").detect.transform("mean")
    dz = (dz / dz.std()).values if dz.std() > 0 else dz.values
    X = np.c_[L, pos, dz]
    for roi in ROIS:
        y = d[roi].values
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        rows.append(dict(subject=s, roi=roi, b_detect=b[-1], b_pos=b[-2], n_runs=len(d), sd_detect_within=float((d.detect - d.groupby("ratio").detect.transform("mean")).std())))
BB = pd.DataFrame(rows)
g1 = []
for roi, x in BB.groupby("roi", sort=False):
    t, p = stats.ttest_1samp(x.b_detect, 0)
    g1.append(dict(roi=roi, n=len(x), mean=x.b_detect.mean(), sem=x.b_detect.std(ddof=1) / np.sqrt(len(x)), t=t, p=p, n_pos=int((x.b_detect > 0).sum())))
G1 = pd.DataFrame(g1)
pd.concat([BB.assign(kind="subject"), G1.assign(kind="group")]).to_csv(os.path.join(OUT, "behavior_brain.csv"), index=False)


# ---------- (4) coupling of residuals ----------
def resid(d, col):
    L, pos = design(d)
    X = np.c_[L, pos]
    y = d[col].values
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


PAIRS = [("Dorsal attention (Wang IPS–FEF)", "Peripheral V1–V3"), ("Dorsal attention (Wang IPS–FEF)", "Foveal V1–V3"),
         ("Dorsal attention (Wang IPS–FEF)", "wang2015|V1_periph"), ("Dorsal attention (Wang IPS–FEF)", "wang2015|V2_periph"),
         ("Dorsal attention (Wang IPS–FEF)", "schaefer2018|DefaultA"), ("Foveal V1–V3", "Peripheral V1–V3")]
rows, pairs = [], []
for s, d in W.groupby("subject"):
    d = d.sort_values("run")
    res = {c: resid(d, c) for c in set(sum(PAIRS, ())) | {"Global (default/control/salience)", "Global (control/salience)"}}
    for a, b in PAIRS:
        gl = res["Global (control/salience)" if "Default" in a + b else "Global (default/control/salience)"]
        r = np.corrcoef(res[a], res[b])[0, 1]
        ra = res[a] - gl * (gl @ res[a]) / (gl @ gl); rb = res[b] - gl * (gl @ res[b]) / (gl @ gl)
        rp = np.corrcoef(ra, rb)[0, 1]
        rows.append(dict(subject=s, a=a, b=b, r=r, r_partial_global=rp, n_runs=len(d)))
        for i in range(len(d)):
            pairs.append(dict(subject=s, a=a, b=b, x=res[a][i], y=res[b][i]))
CC = pd.DataFrame(rows)
g4 = []
for (a, b), x in CC.groupby(["a", "b"], sort=False):
    for col in ["r", "r_partial_global"]:
        z = np.arctanh(x[col].clip(-0.999, 0.999)); t, p = stats.ttest_1samp(z, 0)
        g4.append(dict(a=a, b=b, measure=col, mean_r=float(np.tanh(z.mean())), t=t, p=p, n_neg=int((x[col] < 0).sum()), n=len(x)))
G4 = pd.DataFrame(g4)
pd.concat([CC.assign(kind="subject"), G4.assign(kind="group")]).to_csv(os.path.join(OUT, "coupling.csv"), index=False)
pd.DataFrame(pairs).to_csv(os.path.join(OUT, "coupling_pairs.csv"), index=False)

pd.set_option("display.width", 220)
print("behavior rows matched:", W.detect.notna().sum(), "of", len(W))
print(BB.groupby("subject").sd_detect_within.first().round(1).to_string())
print(G1.round(3).to_string(index=False))
print(G4.round(3).to_string(index=False))
