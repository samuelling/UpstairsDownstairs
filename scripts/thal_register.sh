#!/bin/bash -l
# Nonlinear registration of each subject's anatomy to MNI152 (FSL FLIRT + FNIRT), then the
# Juelich histological LGN/MGN probability maps warped back into the subject's FreeSurfer
# voxel space (orig.mgz grid), for the atlas-agreement reliability measure.
# Reads fs_subjects/<id>/mri (copies); writes thalamus/reg/<id>/. Needs freesurfer + fsl modules.
set -euo pipefail
R=/projectnb/vision/CentralAttentionManipulation_reanalysis
J=$FSLDIR/data/atlases/Juelich/Juelich-prob-1mm.nii.gz
for id in 001 003 004 007 009 011 012 013; do
  d=$R/thalamus/reg/$id; mkdir -p $d
  [ -f $d/juelich_rh_MGN.nii.gz ] && { echo "$id done"; continue; }
  mri_convert $R/fs_subjects/$id/mri/orig.mgz $d/orig.nii.gz >/dev/null
  mri_convert $R/fs_subjects/$id/mri/brainmask.mgz $d/brain.nii.gz >/dev/null
  fslreorient2std $d/orig.nii.gz $d/orig_std.nii.gz; fslreorient2std $d/brain.nii.gz $d/brain_std.nii.gz
  flirt -in $d/brain_std.nii.gz -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain -omat $d/aff.mat -dof 12
  fnirt --in=$d/orig_std.nii.gz --aff=$d/aff.mat --cout=$d/warp --config=T1_2_MNI152_2mm --ref=$FSLDIR/data/standard/MNI152_T1_2mm
  invwarp -w $d/warp -o $d/invwarp -r $d/orig_std.nii.gz
  # index: 102 R LGN, 103 L LGN, 105 R MGN, 106 L MGN (0-based volumes in the 4D file)
  for pair in "rh_LGN 102" "lh_LGN 103" "rh_MGN 105" "lh_MGN 106"; do
    set -- $pair
    fslroi $J $d/mni_$1 $2 1
    applywarp -i $d/mni_$1 -r $d/orig_std.nii.gz -w $d/invwarp -o $d/std_$1 --interp=trilinear
    # back from the reoriented (std) grid to the original orig.mgz voxel grid
    mri_vol2vol --mov $d/std_$1.nii.gz --targ $d/orig.nii.gz --regheader --o $d/juelich_$1.nii.gz --trilin >/dev/null
  done
  echo "$id registered"
done
