# UpstairsDownstairs

**Central load × attention networks: a reanalysis of the NJP central-load fMRI study.**

Does making a central task harder pull attention inward, raising activity in the dorsal attention network and the foveal cortex ("upstairs") while suppressing the periphery ("downstairs")?

**Dashboard:** open [`dashboard/index.html`](dashboard/index.html) in a browser. It has no build step and no dependencies.

## Design
- 8 subjects, about 155 fMRI runs, block design (16 s on / 16 s off, TR 2 s), one load level per run.
- **Central task (NJP):** a ring of red and blue squares at 0.5–1.5°. Subjects report changes in the majority color. Load is the minority proportion: 10, 30, 38, 40, 42, 44 or 46% (near 50% is hard).
- **Periphery:** a task-irrelevant 1/f noise annulus at 3–7.5°, identical in every condition.

## Headline results (all 8 subjects)
| ROI (Wang 2015) | Fitted effect, 10% → 46% (% signal) | t(7) | q (FDR) |
|---|---|---|---|
| IPS0 | +0.82 | 8.1 | .001 |
| IPS1 | +0.91 | 7.7 | .001 |
| IPS2 | +1.16 | 7.1 | .001 |
| IPS3 | +1.01 | 6.0 | .002 |
| IPS4/5 | +0.76 | 4.6 | .005 |
| SPL1 | +0.76 | 6.7 | .001 |
| FEF | +0.51 | 6.9 | .001 |

- Accuracy falls from 95% at the 30% ratio to 41% at the 46% ratio.
- The default mode network decreases with load (Schaefer DefaultA t = −4.2).
- Early visual cortex shows a push-pull: foveal V1–V3 increase, peripheral V1–V3 trend downward (significant only in V2).
- The pattern holds with Glasser HCP-MMP1 parcels and Schaefer 400/17 sub-regions, and when the two subjects with flat behavior (S12, S13) are excluded.

## Caveats
- **Load is confounded with run position.** There is one ratio per run and each subject's order repeats, which the existing GLMs can't separate. Next step: refit with run position as a covariate.
- S12 and S13 show flat behavior across load, even after the original key-reversal correction.
- Eccentricity bands come from the Benson 2014 template, not from subject pRFs.
- Wang IPS4 and IPS5 are merged (IPS5 is only 18–53 vertices at maximum probability).

## Contents
- `dashboard/`: `index.html` and the generated `data.js`.
- `data/`: ROI × ratio betas, per-subject slopes, group stats, and a per-run label/behavior check. These are group- and ROI-level derived values only; subjects appear only by study number.
- `scripts/`
  - `check_par_vs_behavior.py`: checks the CAM.par condition files against the behavioral ratios (run on the cluster).
  - `roi_load_betas.py`: extracts ROI betas from the original FS-FAST fsaverage GLMs using the atlases below (run on the cluster).
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
