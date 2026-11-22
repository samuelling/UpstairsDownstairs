# UpstairsDownstairs

**Central load × attention networks: a reanalysis of the NJP central-load fMRI study.**

Does making a central task harder pull attention inward, raising activity in the dorsal attention network and the foveal cortex ("upstairs") while suppressing the periphery ("downstairs")?

**Dashboard:** open [`dashboard/index.html`](dashboard/index.html) in a browser. It has no build step and no dependencies.

## Design
- 8 subjects, about 155 fMRI runs, block design (16 s on / 16 s off, TR 2 s), one load level per run.
- **Central task (NJP):** a ring of red and blue squares at 0.5–1.5°. Subjects report changes in the majority color. Load is the minority proportion: 10, 30, 38, 40, 42, 44 or 46% (near 50% is hard).
- **Periphery:** a task-irrelevant 1/f noise annulus at 3–7.5°, identical in every condition.

## What's in the dashboard (one TL;DR per analysis)
0. **The task, animated.** A canvas reconstruction of the stimulus built from the original experiment code (`Design/*.m`) and the parameters stored in the behavior files: 173 squares at 0.5–1.5°, reshuffles 25–29 times per 16-s block, a majority-color coin toss every 5–6 reshuffles, a 1.5-s response window, and the 1/f noise annulus at 3–7.5°. You can play along with Blue/Red buttons or the B/R keys and compare your score with the subjects'. A run timeline shows the ON/OFF blocks and each block's reshuffles, flips and response windows, and a gallery shows all seven load levels. The noise updates twice a second by default; real speed (about 30 Hz) is opt-in.
1. **Data and label audit.** All 151 runs are correctly labelled, and raw and preprocessed data exist for every run.
2. **Behavior.** The load manipulation works in all 8 subjects. The task score falls from 95% to 33% between the 30% and 46% ratios, mostly through missed changes: detection falls from 100% to 53%, while accuracy on answered changes falls from 96% to 66%.
3. **Dorsal attention gradient (Wang 2015).** IPS0–IPS4/5, SPL1 and FEF all rise with load (t = 4.6–8.1). IPS2 rises most (+1.16% signal, fitted 10% → 46%).
4. **Across atlases.** The same increase appears with Glasser HCP-MMP1 and Schaefer 400/17 parcels. Default mode A decreases with load (t = −4.2).
5. **Fovea vs periphery, using each subject's own pRF eccentricity.** Foveal V1–V3 rise with load (p = .003–.007). Peripheral V1–V3 trend down, significantly only in V2 (p = .039, not FDR-significant). This matches the template-based split.
6. **Load-function models.** Flat, linear, saturating-exponential and Naka-Rushton models were fit vertex-wise and on ROI averages, then compared with AICc. A load function decisively beats flat in IPS0–IPS4/5, SPL1, FEF, LO and dorsal attention A (summed ΔAICc 33–64). The shape is linear, not compressive, over 10–46%. Simulation shows 7 load levels can detect a load function but can't reliably separate compressive from linear, and never select Naka-Rushton.

7. **Hemispheric effects.** Load effects are larger on the right in the dorsal attention network (R − L +0.32%, t(7) = 2.6, p = .035, 6 of 8 subjects) and the control network (p = .017). This tracks a generally stronger right-hemisphere response, not a right-specific load effect. Default mode shows no asymmetry. In the dashboard, a hemisphere control (Both / Left / Right / Right vs Left) switches sections 3–6, and `?hemi=lh|rh|diff` links straight to a view.

**Headline:** harder central judgments drive the dorsal attention network up, linearly with load and more strongly on the right. Foveal visual cortex rises with load, while the predicted peripheral suppression is weak.

**Correction (2026-11-21):** the first version reported flat behavior for S12 and S13. That was a scoring bug in this reanalysis: after reversing their key mapping, it counted missed changes as correct. Scored properly, they behave like the other subjects. No fMRI result changed.

## Caveats
- **Load is confounded with run position.** There is one ratio per run and each subject's order repeats, which the existing GLMs can't separate. Next step: refit with run position as a covariate.
- At 46% subjects miss about half of the changes, which may mean they disengage; this would explain the dip at 46% in the attention ROIs.
- The S9 eccentricity split uses the Benson 2014 template (that subject has no pRF data). Wang IPS4 and IPS5 are merged.

## Contents
- `dashboard/`: `index.html` and the generated `data.js`.
- `data/`: ROI × ratio betas, per-subject slopes, group stats, and a per-run label/behavior check. These are group- and ROI-level derived values only; subjects appear only by study number.
- `scripts/` (cluster scripts read the original study's files read-only and write to a separate reanalysis folder)
  - `check_par_vs_behavior.py`: audits the condition labels and computes per-run behavior (score, detection, accuracy).
  - `roi_load_betas.py`: atlas ROI betas and slope statistics.
  - `prf_to_fsaverage.sh`: resamples subject pRF eccentricity, R² and V1–V3 labels to fsaverage (FreeSurfer 6).
  - `load_models.py`: fits flat, linear, saturating-exponential and Naka-Rushton models vertex-wise and on ROI averages, with AICc.
  - `hemi_stats.py`: per-hemisphere load slopes, paired right-minus-left tests, per-hemisphere model fits, and right-vs-left composites.
  - `model_recovery_and_prf_stats.py`: model-recovery simulation and pRF-ROI slope statistics.
  - `build_data.py`: builds `dashboard/data.js` from `data/` (stdlib only).

## Atlases (fsaverage)
- Wang, Mruczek, Arcaro & Kastner (2015) *Cereb Cortex*: probabilistic retinotopic ROIs, maximum-probability labels from neuropythy.
- Schaefer et al. (2018) *Cereb Cortex*: 400 parcels, 17 networks (CBIG).
- Glasser et al. (2016) *Nature*: HCP-MMP1.0 projected to fsaverage (figshare 3498446, CC BY 4.0).
- Benson et al. (2014) *PLoS Comput Biol*: retinotopy template (eccentricity).

## Rebuild
```
python3 scripts/build_data.py   # after refreshing data/ from the cluster outputs
```
