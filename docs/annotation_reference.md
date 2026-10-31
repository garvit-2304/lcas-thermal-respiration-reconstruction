# ROI reference annotation

The dataset provides respiration event timestamps and the paper describes a
manually segmented forehead temperature reference, but this workspace has no
pixel-wise face or facial-region masks. The tracker outputs are therefore
candidates, not validation labels.

## Create a reference sample

Run from the project root:

```powershell
python scripts\annotate_roi_reference.py --annotator "Reviewer name"
```

The tool presents four saved preview frames per subject (first, middle,
three-quarter, and final frame). It does not show the existing tracker boxes
while you draw, to reduce anchoring. Drag a rectangle for the selected region;
press `0` for face, `1` forehead, `2` nose, `3` mouth, `4` image-left cheek,
and `5` image-right cheek. Press `x` when a region is not visible, `u` to clear
the selected region, Enter to save and advance, Backspace to go back, `s` to
skip, and `q` to save and quit. Annotations save after every edit and can be
resumed with the same annotator name and output path.

The default output is `roi_reference_review.json`. Each completed frame has a
face box and a reviewed status for all five candidate regions, including
regions marked not visible. Coordinates use original-frame pixel units. The
embedded guide defines the box conventions.

## Compare with the tracker

After completing the sample, run:

```powershell
python scripts\evaluate_roi_reference.py
```

This writes per-box IoU, center error, and area ratio to
`roi_reference_tracker_agreement.csv`, with a summary JSON beside it. Reviewed
boxes are compared even when only some regions on a frame are labeled. The
result is preliminary agreement on this small sample; it is not dataset-wide
accuracy. Have a second reviewer check the boxes before using them as formal
ground truth or selecting a detector.

The selected frames provide a quick check across the static and moving parts
of the recordings. A research-grade validation set should sample more densely
around pose changes and use an independent second reviewer.

## Codex visual draft

`codex_visual_face_box_draft.json` contains rough AI visual estimates for the
face box in frame 1 and the three-quarter preview of each subject. It has no
facial-region labels and is **not** human ground truth. Its agreement reports
are kept separately under `draft_agreement_reference/` and
`draft_agreement_candidate/`; they are useful only for finding gross tracker
failures before a human review.

## Tracker sample feedback

`tracker_sample_review.json` records the project owner's review of the saved
tracker previews. It marks incorrect boxes as failure examples and moving poses
as motion challenges, so those frames are not counted as clean tracker
successes. Jaime frame 2652 is the only Jaime sample currently marked usable by
that review. This feedback assesses the preview boxes and sample suitability;
it does not provide pixel-level face or facial-region labels. The original
images are under `original_dataset/<subject>/`, and the corresponding overlays
are under `data/processed/face_previews/<subject>/`.

## Project-owner reference review

The saved review in `roi_reference_review.json` now has all 20 sampled frames
complete, with the face and five facial regions reviewed on every frame. It
contains 115 visible reference boxes; five regions were explicitly marked not
visible, and no reviewed region is missing a decision. This completes the
annotation of the selected sample, not the full dataset. It remains a
single-annotator reference and needs an independent review before formal use.

Across the 20 face boxes, mean IoU is 0.561 for the baseline tracker, 0.535 for
`min_neighbors=2`, and 0.569 for `scale_factor=1.05, min_neighbors=2`. Across
all 115 visible face and subregion boxes, mean IoU is 0.218, 0.206, and 0.220,
respectively. Region overlap is low, and this sample is too small to select a
tracker. The reproducible per-frame comparison is in
`owner_tracker_variant_comparison.csv` and its summary is in
`owner_tracker_variant_comparison.json`.

`roi_geometry_loso_comparison.json` compares fixed ROI ratios with median
ratios learned from four subjects and evaluated on the held-out fifth. With
manually annotated face boxes, overall mean IoU changes from 0.387 to 0.416;
with baseline tracker boxes it changes from 0.145 to 0.087. This indicates
that face-box localization is a major source of error and that the learned
ratios do not improve the current tracker pipeline. No geometry variant is
selected for production. The per-box results are in
`roi_geometry_loso_comparison.csv`.

The manually boxed nose and other facial regions provide reference only for
these 20 selected frames. They do not form continuous full-recording tracks
and cannot be substituted directly as the TCN's full-sequence input. The
region/graph rate ablation is reported separately under
`data/processed/region_graph_ablations/` and is explicitly exploratory.

