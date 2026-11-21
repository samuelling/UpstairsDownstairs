#!/bin/bash -l
# Resample each subject's pRF eccentricity + R2 maps and pRF V1-V3 labels to
# fsaverage (FreeSurfer 6.0, same commands as Sara's RegisterPrfEccns.txt /
# registerlabels). Reads /project/vision only; writes only to
# $ROOT/prf_fsaverage. Subject 009 has no pRF data (template fallback).
module load freesurfer/6.0 >/dev/null 2>&1   # before set -u: the module function trips on unset vars
set -euo pipefail
command -v mri_surf2surf >/dev/null
export SUBJECTS_DIR=/project/vision/subjects
ROOT=/projectnb/vision/CentralAttentionManipulation_reanalysis
OUT=$ROOT/prf_fsaverage
mkdir -p "$OUT"
for id in 001 003 004 007 011 012 013; do
  for hemi in lh rh; do
    for m in ecc R2; do
      mri_surf2surf --srcsubject "$id" --trgsubject fsaverage --hemi "$hemi" \
        --sval "/project/vision/pRF_Retinotopy/$id/pRF_surfoverlays/$hemi.pRF_$m.nii.gz" \
        --tval "$OUT/$hemi.S$id.pRF_$m.fsaverage.mgz" > "$OUT/log.$hemi.S$id.$m.txt" 2>&1
    done
    for roi in v1 v2 v3; do
      mri_label2label --srcsubject "$id" --trgsubject fsaverage --hemi "$hemi" --regmethod surface \
        --srclabel "$SUBJECTS_DIR/$id/label/pRF_retinotopy/$hemi.pRF.$roi.label" \
        --trglabel "$OUT/$hemi.S$id.pRF.$roi.fsaverage.label" > "$OUT/log.$hemi.S$id.$roi.txt" 2>&1
    done
  done
  echo "done S$id"
done
