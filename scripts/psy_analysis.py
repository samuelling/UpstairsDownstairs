#!/usr/bin/env python3
"""Reanalysis of the AttOD dual-task psychophysics (AttentionManipulation).

Every trial (50 ms) shows 200 red/blue squares around fixation (central
numerosity task: which color is the majority?) and a full-field 2 cpd, 40%
contrast grating with a central hole, tilted left/right of vertical
(peripheral orientation-discrimination task). Subjects report both. Tilt is
set by QUEST (80% correct), with one staircase per blue-square count; each run
interleaves 2 minority proportions x 2 majority colors x 40 trials.

Source data are read-only. Subjects are coded P01-P10 (folder order); the
name key is written only to the cluster folder, never to outputs/.

Outputs (../outputs, all subject-coded):
  psy_cells.csv         subject x ratio: central accuracy/response rate, OD accuracy,
                        QUEST threshold (mean of run-level modes), psychometric threshold
  psy_runs.csv          subject x run x condition QUEST thresholds and accuracies
  psy_group.csv         group mean +/- SEM at the 8 common ratios (several measures)
  psy_slopes.csv        per-subject slopes vs ratio + group t-tests
  psy_trial_glm.csv     per-subject logistic regression of OD correctness
  psy_original.csv      original criteria (exclusions, 7-subject mean), for comparison
  psy_psychometric.csv  pooled psychometric curves (binned) per ratio, for plotting
"""
import glob, os
import numpy as np, pandas as pd, scipy.io as sio
from scipy import stats, optimize
from scipy.special import expit

SRC = "/projectnb/vision/AttentionManipulation"
ROOT = "/projectnb/vision/AttentionManipulation_reanalysis"
OUT = f"{ROOT}/outputs"
os.makedirs(OUT, exist_ok=True)
L = lambda f: sio.loadmat(f, squeeze_me=True, struct_as_record=False)
NTOT = 200
COMMON = [0.18, 0.28, 0.36, 0.40, 0.42, 0.44, 0.46, 0.48]
GAMMA, DELTA = 0.5, 0.01

dirs = sorted(d for d in glob.glob(f"{SRC}/DataSbj*AttOD") if os.path.isdir(d))
key = pd.DataFrame({"code": [f"P{i + 1:02d}" for i in range(len(dirs))], "folder": [os.path.basename(d) for d in dirs]})
key.to_csv(f"{ROOT}/subject_key_DO_NOT_SHARE.csv", index=False)

trials, runs = [], []
for code, d in zip(key.code, dirs):
    fs = glob.glob(f"{d}/Data_AttOD_Run*.mat")
    fs = sorted(fs, key=lambda f: int(os.path.basename(f).split("_Run")[1].split("_")[0]))
    for f in fs:
        run = int(os.path.basename(f).split("_Run")[1].split("_")[0])
        m = L(f); r = m["results"]; e = m["events"]
        nb = np.atleast_1d(e.numGreens).astype(int)
        cvec = np.atleast_1d(m["rects"].nGreenVec).astype(int)
        df = pd.DataFrame({
            "subject": code, "run": run, "nblue": nb,
            "tilt": np.atleast_1d(r.tiltVec).astype(float),
            "od_correct": np.atleast_1d(r.correctRespOD).astype(int),
            "rb_correct": np.atleast_1d(r.correctRespRB).astype(int),
            "rb_resp": ~np.isnan(np.atleast_1d(r.respRBTask).astype(float)),
            "od_resp": ~np.isnan(np.atleast_1d(r.respODTask).astype(float)),
        })
        df["trial"] = np.arange(len(df))
        trials.append(df)
        est = 10 ** np.atleast_1d(r.finalEstimate).astype(float)
        for c, th in zip(cvec, est):
            sub = df[df.nblue == c]
            runs.append(dict(subject=code, run=run, nblue=c, quest_thr=th,
                             rb_acc=sub.rb_correct.mean(), od_acc=sub.od_correct.mean()))
T = pd.concat(trials, ignore_index=True)
R = pd.DataFrame(runs)
for X in (T, R):
    X["ratio"] = (np.minimum(X.nblue, NTOT - X.nblue) / NTOT).round(2)
    X["blue_majority"] = X.nblue > NTOT / 2
R.to_csv(f"{OUT}/psy_runs.csv", index=False)


# ---------- psychometric fits (Weibull, guess .5, lapse .01; slope shared within subject) ----------
def pcorr(x, a, b, lam=DELTA):
    return GAMMA + (1 - GAMMA - lam) * (1 - np.exp(-(x / a) ** b))

def fit_subject(sub):
    rs = sorted(sub.ratio.unique())
    x = [sub.tilt[sub.ratio == q].values for q in rs]
    y = [sub.od_correct[sub.ratio == q].values for q in rs]
    def nll(p):
        b = np.exp(p[0]); lam = p[-1]; out = 0.0
        for i in range(len(rs)):
            pc = np.clip(pcorr(x[i], np.exp(p[1 + i]), b, lam), 1e-6, 1 - 1e-6)
            out -= np.sum(y[i] * np.log(pc) + (1 - y[i]) * np.log(1 - pc))
        return out
    p0 = np.r_[np.log(2.0), np.log([max(0.5, np.median(xi)) for xi in x]), 0.05]
    res = optimize.minimize(nll, p0, method="L-BFGS-B", bounds=[(np.log(0.5), np.log(10))] + [(np.log(0.05), np.log(60))] * len(rs) + [(0.0, 0.3)])
    b = np.exp(res.x[0]); a = np.exp(res.x[1:-1]); lam = res.x[-1]
    # threshold = alpha (63% point of the Weibull, independent of lapse); thr75 = 75% of the way between guess and asymptote
    return pd.DataFrame({"ratio": rs, "pf_thr": a, "pf_beta": b, "pf_lapse": lam})

cells = []
for s, sub in T.groupby("subject"):
    pf = fit_subject(sub)
    agg = sub.groupby("ratio").agg(n_trials=("trial", "size"), rb_acc=("rb_correct", "mean"), rb_resp=("rb_resp", "mean"),
                                   od_acc=("od_correct", "mean"), od_resp=("od_resp", "mean"), median_tilt=("tilt", "median")).reset_index()
    q = R[R.subject == s].groupby("ratio").agg(quest_thr=("quest_thr", "mean"), quest_thr_median=("quest_thr", "median"), n_staircases=("quest_thr", "size")).reset_index()
    cells.append(agg.merge(q, on="ratio").merge(pf, on="ratio").assign(subject=s))
C = pd.concat(cells, ignore_index=True)
C["common"] = C.ratio.isin(COMMON)
C.to_csv(f"{OUT}/psy_cells.csv", index=False)

MEAS = {"rb_acc": 100, "rb_resp": 100, "od_acc": 100}
LOGMEAS = ["quest_thr_median", "pf_thr"]

# ---------- reliability (original criteria): run x count cells valid if central acc >= .5 and QUEST
#            threshold <= 20 deg; a subject is reliable if > 70% of cells are valid ----------
R["valid"] = (R.rb_acc >= 0.5) & (R.quest_thr <= 20)
PV = R.groupby("subject").valid.mean()
RELIABLE = sorted(PV[PV > 0.70].index)
C["reliable"] = C.subject.isin(RELIABLE)
C.to_csv(f"{OUT}/psy_cells.csv", index=False)
SETS = {"reliable": RELIABLE, "all": sorted(C.subject.unique())}

# ---------- group curves at the common ratios (thresholds: geometric mean, SEM in log units) ----------
g = []
for setname, subs in SETS.items():
    for q in COMMON:
        x = C[(C.ratio == q) & C.subject.isin(subs)]
        row = {"set": setname, "ratio": q, "n": len(x)}
        for k, sc in MEAS.items():
            v = x[k].values * sc; row[f"{k}_m"] = v.mean(); row[f"{k}_se"] = v.std(ddof=1) / np.sqrt(len(v))
        for k in LOGMEAS:
            v = np.log10(x[k].values); m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
            row[f"{k}_gm"] = 10 ** m; row[f"{k}_lo"] = 10 ** (m - se); row[f"{k}_hi"] = 10 ** (m + se)
        g.append(row)
GR = pd.DataFrame(g); GR.to_csv(f"{OUT}/psy_group.csv", index=False)

# ---------- per-subject slopes vs ratio (all tested ratios); thresholds in log10 deg ----------
srows = []
for s, x in C.groupby("subject"):
    row = {"subject": s, "n_ratios": len(x), "reliable": s in RELIABLE, "pct_valid": PV[s], "pf_lapse": x.pf_lapse.iloc[0], "pf_beta": x.pf_beta.iloc[0]}
    row["slope_rb_acc"] = np.polyfit(x.ratio, 100 * x.rb_acc, 1)[0] * 0.30          # change in % from 18% to 48%
    row["slope_rb_resp"] = np.polyfit(x.ratio, 100 * x.rb_resp, 1)[0] * 0.30
    for k in LOGMEAS:
        row[f"slope_log_{k}"] = np.polyfit(x.ratio, np.log10(x[k]), 1)[0] * 0.30     # log10 change, 18% -> 48%
    rr = R[R.subject == s]
    A = np.c_[np.ones(len(rr)), rr.ratio, rr.run]
    coef = np.linalg.lstsq(A, np.log10(rr.quest_thr.values), rcond=None)[0]
    row["slope_log_quest_ctrl_run"] = coef[1] * 0.30; row["run_slope_log_quest"] = coef[2]
    e = x[x.ratio <= 0.36]; h = x[x.ratio >= 0.44]
    row["easy_hard_ratio_quest"] = 10 ** (np.log10(h.quest_thr_median).mean() - np.log10(e.quest_thr_median).mean())
    row["easy_hard_ratio_pf"] = 10 ** (np.log10(h.pf_thr).mean() - np.log10(e.pf_thr).mean())
    srows.append(row)
S = pd.DataFrame(srows)
tests = []
for setname, subs in SETS.items():
    Ss = S[S.subject.isin(subs)]
    for k in [c for c in S.columns if c.startswith(("slope_", "run_slope"))] + ["log_easy_hard_quest", "log_easy_hard_pf"]:
        v = np.log10(Ss["easy_hard_ratio_" + k.split("_")[-1]]).values if k.startswith("log_easy") else Ss[k].values
        t, p = stats.ttest_1samp(v, 0); w = stats.wilcoxon(v).pvalue
        tests.append(dict(set=setname, measure=k, mean=v.mean(), sem=v.std(ddof=1) / np.sqrt(len(v)), t=t, p=p, wilcoxon_p=w, n_pos=int((v > 0).sum()), n=len(v)))
    rr_ = stats.spearmanr(Ss.slope_rb_acc, Ss.slope_log_quest_thr_median)
    tests.append(dict(set=setname, measure="spearman_central_drop_vs_threshold_rise", mean=rr_.statistic, sem=np.nan, t=np.nan, p=rr_.pvalue, wilcoxon_p=np.nan, n_pos=np.nan, n=len(Ss)))
S.to_csv(f"{OUT}/psy_subjects.csv", index=False)
TS = pd.DataFrame(tests); TS.to_csv(f"{OUT}/psy_slopes.csv", index=False)

# ---------- trial-level: OD correct ~ log tilt + ratio + central correct (per subject logistic) ----------
glm = []
for s, x in T.groupby("subject"):
    X = np.c_[np.ones(len(x)), np.log10(x.tilt.clip(0.05)), (x.ratio - 0.33) / 0.15, x.rb_correct]
    y = x.od_correct.values
    nll = lambda b: -np.sum(y * (X @ b) - np.logaddexp(0, X @ b))
    b = optimize.minimize(nll, np.zeros(4), method="BFGS").x
    glm.append(dict(subject=s, b0=b[0], b_logtilt=b[1], b_ratio=b[2], b_central_correct=b[3],
                    od_acc_central_correct=x.od_correct[x.rb_correct == 1].mean(), od_acc_central_wrong=x.od_correct[x.rb_correct == 0].mean()))
G = pd.DataFrame(glm)
G["reliable"] = G.subject.isin(RELIABLE)
G.to_csv(f"{OUT}/psy_trial_glm.csv", index=False)
gt = []
for setname, subs in SETS.items():
    Gs = G[G.subject.isin(subs)]
    for k in ["b_logtilt", "b_ratio", "b_central_correct"]:
        t, p = stats.ttest_1samp(Gs[k], 0); gt.append(dict(set=setname, term=k, mean=Gs[k].mean(), sem=Gs[k].std(ddof=1) / np.sqrt(len(Gs)), t=t, p=p, n_pos=int((Gs[k] > 0).sum()), n=len(Gs)))
pd.DataFrame(gt).to_csv(f"{OUT}/psy_trial_glm_group.csv", index=False)

# ---------- original pipeline (MeanPerfThrshAcrossSbjs.m): mean of valid run-level QUEST thresholds ----------
o = R[R.valid & R.subject.isin(RELIABLE)].groupby(["subject", "ratio"]).agg(thr=("quest_thr", "mean"), rb=("rb_acc", "mean")).reset_index()
orow = []
for q in COMMON:
    x = o[o.ratio == q]
    orow.append(dict(ratio=q, n=len(x), thr_m=x.thr.mean(), thr_se=x.thr.std(ddof=1) / np.sqrt(len(x)), rb_m=100 * x.rb.mean(), rb_se=100 * x.rb.std(ddof=1) / np.sqrt(len(x))))
pd.DataFrame(orow).to_csv(f"{OUT}/psy_original.csv", index=False)

# ---------- pooled psychometric data for plotting (log-tilt bins, easy vs hard) ----------
T["load_bin"] = np.where(T.ratio <= 0.36, "easy (18–36%)", np.where(T.ratio >= 0.44, "hard (44–48%)", "middle"))
edges = np.logspace(np.log10(0.3), np.log10(30), 11)
T["tbin"] = pd.cut(T.tilt, edges)
pm = T[(T.load_bin != "middle") & T.subject.isin(RELIABLE)].groupby(["load_bin", "subject", "tbin"], observed=True).od_correct.mean().reset_index()
pm = pm.groupby(["load_bin", "tbin"], observed=True).od_correct.agg(["mean", "sem", "count"]).reset_index()
pm["tilt_center"] = [np.sqrt(iv.left * iv.right) for iv in pm.tbin]
pm.drop(columns="tbin").to_csv(f"{OUT}/psy_psychometric.csv", index=False)

print("subjects", len(key), "runs", R[["subject", "run"]].drop_duplicates().shape[0], "trials", len(T), "reliable", RELIABLE)
pd.set_option("display.width", 250)
print(GR[["set", "ratio", "n", "rb_acc_m", "rb_resp_m", "od_acc_m", "quest_thr_median_gm", "pf_thr_gm"]].round(2).to_string(index=False))
print(S.round(3).to_string(index=False))
print(TS.round(4).to_string(index=False))
print(pd.DataFrame(gt).round(4).to_string(index=False))
print(pd.DataFrame(orow).round(2).to_string(index=False))
