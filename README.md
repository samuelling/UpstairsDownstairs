# UpstairsDownstairs

**Central load × attention networks: a reanalysis of the NJP central-load fMRI study.**

Does making a central task harder pull attention inward, raising activity in the dorsal attention network and the foveal cortex ("upstairs") while suppressing the periphery ("downstairs")?

**Dashboard:** open [`dashboard/index.html`](dashboard/index.html) in a browser. It has no build step and no dependencies.

## Design
- 8 subjects, about 155 fMRI runs, block design (16 s on / 16 s off, TR 2 s), one load level per run.
- **Central task (NJP):** a ring of red and blue squares at 0.5–1.5°. Subjects report changes in the majority color. Load is the minority proportion: 10, 30, 38, 40, 42, 44 or 46% (near 50% is hard).
- **Periphery:** a task-irrelevant 1/f noise annulus at 3–7.5°, identical in every condition.

## How the dashboard is organized
The page reads as a presentation of the results, not a log of analyses. It opens with five takeaways and the key numbers. **Part one, The experiment**, has the animated task (01) and the manipulation check (02). **Part two, What the brain did**, covers the attention network (03, with an across-atlas check), fovea vs periphery (04), the shape of the load function (05) and hemispheres (06). Then come **The verdict** (07: interpretation, open issues, next steps) and an appendix with methods, data checks and the S12/S13 correction. Each finding starts with a takeaway; the method and tables are in a collapsible "How this was tested" drawer.

## Analyses (one TL;DR each)
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

## Psychophysics tab (`dashboard/psychophysics.html`)
A reanalysis of the dual-task psychophysics study that preceded the fMRI (cluster source: `AttentionManipulation`, read-only; outputs in `AttentionManipulation_reanalysis`). There are 10 subjects (coded P01–P10), 300 runs and 48,000 trials. Every 50-ms trial showed 200 red/blue squares at fixation (majority-color judgment, minority share 18–48%) together with a peripheral 2 cpd grating (tilt left or right). QUEST measured the tilt threshold at each central load.
- **Central accuracy** falls from 97% to 61% across loads (t(6) = −9.8).
- **Peripheral tilt thresholds** rise modestly with load: about 1.3× from easy (18–36%) to hard (44–48%) loads, in 6 of 7 reliable subjects, p = .08. Three subjects carry most of the effect, and thresholds dip at 48%.
- **Trial by trial**, holding tilt constant, a tilt judgment is more often correct when the central judgment on that trial was correct (7 of 7 subjects, p = .018).
- **Reliability:** following the original criteria, the main analyses use the 7 subjects with more than 70% usable staircases; all 10 are shown as a check.

- **Contrast-discrimination pilots (section 06):** TvC functions from 2 pilot subjects (coded A and B). A Naka-Rushton response-gain change explains load better than contrast gain: Rmax ×0.76 from 5% to 48% load in both subjects, ×0.52 across 12–48% in B. Code: `scripts/rsf_extract.py` (cluster) and `scripts/rsf_tvc.py` (local).

Rebuild: `rsync -a scc:/projectnb/vision/AttentionManipulation_reanalysis/outputs/ data/psychophysics/ && python3 scripts/build_psy_data.py`.

## Caveats
- **Run position (now controlled).** There is one ratio per run in a repeating order. A run-by-run refit (`scripts/glm_runpos.py`, a cluster batch job) with run position, scan day and motion as covariates reproduces the original slopes (r = .99) and leaves the load effects essentially unchanged (IPS2 +0.90 → +0.87% fitted). No ROI's run-position term survives FDR. Peripheral V2 suppression becomes p = .024.
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
  - `psy_analysis.py`: the psychophysics reanalysis (cluster).
  - `build_psy_data.py`: builds `dashboard/psy_data.js` from `data/psychophysics/`.
- `dashboard/style.css`: styles shared by both tabs.

## Atlases (fsaverage)
- Wang, Mruczek, Arcaro & Kastner (2015) *Cereb Cortex*: probabilistic retinotopic ROIs, maximum-probability labels from neuropythy.
- Schaefer et al. (2018) *Cereb Cortex*: 400 parcels, 17 networks (CBIG).
- Glasser et al. (2016) *Nature*: HCP-MMP1.0 projected to fsaverage (figshare 3498446, CC BY 4.0).
- Benson et al. (2014) *PLoS Comput Biol*: retinotopy template (eccentricity).

## Rebuild
```
python3 scripts/build_data.py   # after refreshing data/ from the cluster outputs
```
- **Attentional response function (`scripts/arf_demand.py`).** Power-law fits against three load axes: minority share (q ≈ 1.1–1.9, linear), evidence margin on a log axis (q ≈ 0.5–0.8, not different from linear) and behavioral difficulty (q ≈ 0.2–0.35, compressive but poorly fitting). There is no evidence of saturation within 10–46%.
- Rebuild these two sections: `python3 scripts/build_extra_data.py` → `dashboard/extra_data.js`.

## Eccentricity, localizer and run-level analyses
- **Eccentricity profile (section 04b, `scripts/ecc_localizer.py`, cluster):** the load effect in V1–V3 by pRF eccentricity is positive in the fovea (+0.58% at 0.5–0.75°), crosses zero near 2.5°, and is negative from 3° out, reaching −0.41% at 6–7.5° (p = .032) and −0.34% at 9–11° (p = .027). It extends beyond the annulus, so it is a broad peripheral decrease rather than suppression of the ignored stimulus.
- **Annulus localizer (04b):** annulus-driven V2 decreases (−0.37%, p = .046). The strongest-driven vertices decrease most, but that is confounded with eccentricity.
- **Run level (section 06b, `scripts/run_level_coupling.py`):** within a load level, run-to-run detection does not predict dorsal attention activity. After removing shared fluctuations, dorsal attention is still positively coupled with peripheral V1–V3 (no push-pull) and trends negative with the default network (r = −.28, p = .052).

## Brain maps tab (`dashboard/maps.html`)
An interactive inflated-cortex viewer (three.js) showing every ROI used in the analyses (Wang areas, V1–V3 fovea/periphery bands, Schaefer networks, Glasser areas), colored by region or by its group load effect. It also shows the vertex-wise group load-effect map, optionally limited to p < .05. Surfaces are fsaverage6 (nested in fsaverage, so ROIs and maps transfer exactly). Build: `scripts/brain_export.py` on the cluster (with `module load freesurfer/6.0`), then copy `outputs/brain/` to `dashboard/brain/`. It must be served over HTTP (GitHub Pages or `python -m http.server`), not opened as a local file.
- **Volume view (Brain maps tab):** three linked slices (sagittal, coronal, axial) through the MNI305 2 mm template. Cortex shows the 8-subject surface map projected into the cortical ribbon; subcortex and cerebellum come from the original volumetric GLM (6 subjects; S12 and S13 have none). Build: `scripts/vol_export.py` on the cluster, which writes `dashboard/brain/vol.bin` and `vol.json`.
- **Tab order:** fMRI, Brain maps, Psychophysics.
