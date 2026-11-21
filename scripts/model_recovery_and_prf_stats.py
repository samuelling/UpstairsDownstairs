#!/usr/bin/env python3
"""(1) Model-recovery simulation for the AICc load-function comparison
(N = 7 load levels, amplitude 1 %SC, Gaussian noise) -> outputs/model_recovery.csv
(2) Group slope stats for subject-pRF ROIs (same format as group_slopes.csv)
    -> outputs/prf_group_slopes.csv"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_models as L
import roi_load_betas as base

rng = np.random.default_rng(0)
X, V = L.X, 4000
truth = {"flat": np.zeros(7), "linear": X, "exp": (1 - np.exp(-X / 0.2)) / (1 - np.exp(-1 / 0.2)),
         "NR": X ** 3 / (X ** 3 + 0.4 ** 3)}
rows = []
for noise in [0.1, 0.25, 0.5]:
    for name, mu in truth.items():
        b, _ = L.best_model(L.fit_all(mu[:, None] + rng.normal(0, noise, (7, V))))
        rows.append(dict(noise_sd=noise, truth=name, **{f"pct_{m}": 100 * float(np.mean(b == i)) for i, m in enumerate(L.MODELS)}))
pd.DataFrame(rows).to_csv(f"{base.OUT}/model_recovery.csv", index=False)

d = pd.read_csv(f"{base.OUT}/roi_betas_long_prf.csv")
bh = d.groupby(["subject", "roi", "ratio"], as_index=False).beta.mean()
out = []
for roi, g in bh.groupby("roi"):
    sl, rho, bm = [], [], []
    for s, x in g.groupby("subject"):
        x = x.sort_values("ratio")
        sl.append(np.polyfit(x.ratio, x.beta, 1)[0]); rho.append(stats.spearmanr(x.ratio, x.beta).statistic); bm.append(x.beta.mean())
    sl = np.array(sl); t, p = stats.ttest_1samp(sl, 0)
    out.append(dict(subjects="all8", atlas="prf", roi=roi, n=len(sl), mean_slope=sl.mean(), sem=sl.std(ddof=1) / np.sqrt(len(sl)),
                    t=t, p=p, mean_rho=np.mean(rho), mean_beta=np.mean(bm)))
o = pd.DataFrame(out); o["q_fdr_within_atlas"] = base.fdr(o.p.values)
o.to_csv(f"{base.OUT}/prf_group_slopes.csv", index=False)
print(o[["roi", "mean_slope", "t", "p", "q_fdr_within_atlas"]].round(3).to_string(index=False))
