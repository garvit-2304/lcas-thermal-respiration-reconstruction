# L-CAS data provenance and preprocessing

`source/lcas_ground_truth_official_2020-08-03.zip` is an archived copy of the
ground-truth ZIP linked from the official [L-CAS dataset page](https://lcas.lincoln.ac.uk/wp/research/data-sets-software/lcas-thermal-physiological-monitoring-dataset/).
The currently published Nextcloud download links returned HTTP 404 during this
retrieval, so the Internet Archive snapshot of the official download endpoint
was used. The ZIP contains the ten heartbeat and respiration CSVs already in
`../original_dataset/`; all ten files were compared by SHA-256 and match exactly.

Run `python scripts/preprocess_lcas.py` from the project root to create:

- `processed/physiology_long.csv`: timestamp-preserving long-form data, with a
  per-recording relative-time column and separate streams.
- `processed/preprocessing_report.json`: source hashes, per-stream timing and
  value summaries, local-file comparisons, and stream-overlap information.

The preprocessing does not interpolate or resample. Respiration annotation
codes are retained as raw values; the meaning of code `2` is not inferred.

Thermal PNG frames are present under `../original_dataset/<subject>/frame_*.png`: Anestis 2,863; Claudio 2,903; Jaime 2,652; Joao 3,036; and Manuel 2,700. They decode as 382 × 288, 8-bit RGB images with replicated grayscale channels. Treat them as image intensities, not temperatures. The five original ROS bags under `../ros/` contain matching `mono16` sensor matrices and image-header timestamps. Recovered raw counts are stored unchanged in compressed HDF5 files under `processed/raw_thermal/`. The official L-CAS page documents conversion to Celsius as `(count - 1000) / 10`; this conversion is applied to derived signal columns, not to the stored raw matrices.

Frame filenames contain sequence indices, but bag messages provide capture timestamps. `scripts/audit_rosbag_timestamps.py` finds 14,154 thermal image messages, exactly matching all PNG sequence counts. `scripts/synchronize_rosbag_frames.py` maps bag order to one-based frame indices and compares image-header times with physiology timestamps. Four count-matched recordings agree at the same row index with median absolute offsets of 0.5–0.6 ms; Anestis has 3,526 reference samples for 2,863 images and is assessed by nearest timestamps. The source time zone is unspecified; interpreting timestamps as UTC is supported by their close agreement with ROS Unix times.

Run `python scripts/audit_thermal_frames.py` to create:

- `processed/thermal_frames_manifest.csv`: one row per frame with its source
  path, encoding, and basic intensity-quality statistics.
- `processed/frame_alignment_audit.csv`: frame/reference count comparison per
  subject, without silently joining unmatched or untimed samples.
- `processed/thermal_frame_audit.json`: sequence, decode, encoding, and timing
  caveats.

Run `python scripts/detect_track_regions.py` to create a face-box and ROI
baseline:

- `processed/face_tracks.csv`: one face box and five geometric candidate ROI
  rectangles per frame, with detector/tracking status.
- `processed/face_tracking_report.json`: coverage, status counts, method, and
  limitations.
- `processed/face_previews/`: overlays for the first, middle, later-motion, and
  final frames of each subject.

The baseline generated candidate boxes for all 14,154 frames. This is coverage
from a heuristic tracker, not measured localization accuracy. The ROI
rectangles still need validation against landmarks or manual masks.

The face detector accepts `--min-neighbors` and `--scale-factor` for isolated
parameter comparisons. A candidate run with `--min-neighbors 2` and
`--scale-factor 1.05` is stored under `processed/face_tracking_scale_105_neighbors_2/`, with
its own tracks, previews, extracted signals, quality audit, and baseline. Its
comparison with the original run is in
`processed/face_tracking_scale_105_neighbors_2/tracking_variant_comparison.csv`
and `.json`. The candidate fixes Jaime's initial face box but does not track
head turns reliably, leaves eight Anestis frames without an ROI, and changes
three subjects' spectral rates. It remains an experiment. The owner has now
completed the 20-frame face and facial-region reference sample (115 visible
boxes, five regions marked not visible). Mean IoU across all visible boxes is
0.218 for the original tracker, versus 0.206 and 0.220 for the two candidates.
The sample is single-annotator and preliminary, so no tracker is selected from
it. See `annotations/owner_tracker_variant_comparison.json` and
`annotations/roi_geometry_loso_comparison.json` for tracker and ROI geometry
comparisons.

`scripts/evaluate_region_graph_ablations.py` compares raw, PWS, and GTV signals
for all five candidate regions against the mapped inhalation intervals, using
the same fixed 10–40 BPM estimator for each subject. Across five subjects, the
mouth signal has the lowest exploratory rate MAE (2.51 BPM against event-span
rates); graph filtering does not improve that result. This does not select a
production ROI because the numeric annotation code map and full-recording ROI
tracks are not independently verified. Results are in
`processed/region_graph_ablations/`.

`scripts/evaluate_event_interval_rates.py` adds rate errors for the held-out
predictions from the sparse-event TCN evaluation. The TCN predicts too few
events (macro F1 0.141; mean event-span rate MAE 13.42 BPM); the nonlearned
bandpass peak/valley comparator has macro F1 0.318 and mean event-span rate
MAE 1.53 BPM. Its count-derived rate can agree despite event timing errors, so
rate and event metrics are reported separately. See
`processed/tcn_inhalation_event_detector/tcn_event_interval_rate_comparison.json`.

Run `python scripts/extract_rosbag_thermal_matrices.py` to extract raw matrices and timestamps into compressed HDF5 files. Run `python scripts/extract_raw_roi_signals.py` to create timestamped raw-count and derived Celsius signals using the existing PNG-derived face and region rectangles. Each row retains its image-header time. Sampled first, middle, and last raw/PNG frame pairs have pixel correlations above 0.99998. Run `python scripts/plot_timestamped_roi_signals.py` to create five time-axis ROI and filtered-nose plots under `processed/roi_signal_timestamp_plots/`.

Run `python scripts/extract_roi_signals.py` to create `processed/roi_signals.csv` and `processed/roi_signal_report.json`. These contain per-frame face and five-region means/medians from the PNG images. The rows use frame indices; timestamped raw sensor signals are in `processed/roi_signals_raw_mono16.csv`.

Run `python scripts/plot_roi_signals.py` to create one six-panel PNG-intensity plot per subject under `processed/roi_signal_plots/`. These remain frame-domain visualizations; timestamped raw mono16 regional signals and plots are stored separately.

The original L-CAS paper reports 27 Hz, while the official dataset page also lists the Optris PI-450 frame rate as 10 Hz. The bags resolve timing for local images: median image intervals are about 37–38 ms, and irregular gaps produce effective full-recording rates from 21.8 to 24.9 Hz. The official ROS node timestamps processed ROI samples with its local elapsed clock, linearly resamples them, and searches 10–40 BPM.

The project-owner-confirmed working mapping is `0 = no event`, `1 = inhalation`, and `2 = exhalation`. The paper describes manually annotated inhalation instants, and code 1 appears sparsely at roughly 18–22 events per minute across recordings. Code 2 appears only in Joao and alternates with code 1. The dataset authors do not publish a numeric codebook, so this project mapping is documented as a working convention rather than an author-verified definition. See `scripts/audit_respiration_annotation_codes.py` and `processed/respiration_annotation_code_audit.json` for counts and mapped intervals.

The code audit records the official L-CAS page, Cosar et al. (2018), and the
LCAS ROS implementation as reviewed sources. None maps CSV values to event
types; the project owner confirmed the working mapping above. The official
page's current Nextcloud ground-truth link returned 404 during the check. Raw
numeric codes remain preserved, with mapped labels added as a separate column
in `processed/physiology_long.csv`.

Run `python scripts/compare_provisional_code1_event_rates.py` to create
`processed/provisional_code1_event_rate_comparison.csv` and `.json`. The output
compares a full-recording spectral rate with inhalation-event intervals under
the project-owner-confirmed working mapping. It is descriptive only and does
not report event-level accuracy; code 2 is available only for Joao.

Run `python scripts/compare_phase_specific_respiration_rates.py` to compare
separate 0–60 s static and 60–120 s head-movement windows. Against the mapped
inhalation event-span rate, current mean absolute differences are 4.13 BPM in
the static minute and 6.59 BPM in the movement minute; the thermal estimate is
below the reference in all ten windows. This is descriptive rate agreement,
not event detection accuracy. The movement windows are particularly sensitive
to the known tracker/ROI failures, and the annotation codebook is not
author-documented.

For face/ROI reference labels, follow `data/annotations/README.md`. A ten-frame
Codex visual face-box draft and separate tracker-agreement diagnostics are
stored under `data/annotations/`; they are explicitly non-ground-truth and do
not replace the human annotation step.

Owner feedback on the saved tracker samples is recorded in
`data/annotations/tracker_sample_review.json`. Reported bad boxes and moving
poses are retained as failure/challenge cases and are not clean-success
examples.

The generic configurable baseline is implemented in `scripts/estimate_respiration_baseline.py`. It requires explicit `--fps`, `--fps-source`, `--low-hz`, and `--high-hz` inputs. The measured-timestamp run uses `scripts/estimate_rosbag_timestamp_baseline.py` and the raw signals described above; it does not need a nominal frame-rate assumption.

Run `python scripts/audit_timestamped_roi_quality.py` to create per-frame
face-box movement and nose-signal jump metrics under
`processed/roi_quality_audit/`. The script flags within-subject 99th-percentile
outliers for review but does not remove frames. João's visible nose-region
change around 90–101 seconds coincides with face-box movement and raw-signal
jumps. All five plots have been reviewed; each has a small number of large
motion or signal excursions, so tracker/ROI validation and motion-aware
estimation remain open.

`scripts/estimate_candidate_aligned_baseline.py` and its outputs are retained as an earlier provisional ordinal-alignment result. It covered Claudio, Jaime, Joao, and Manuel and excluded Anestis. It is superseded by `scripts/estimate_rosbag_timestamp_baseline.py`, which uses recovered image-header timestamps and raw mono16 nose-region signals for all five recordings. The 10–40 BPM search-band estimates are Anestis 19.23, Claudio 12.74, Jaime 16.79, Joao 14.27, and Manuel 11.79 BPM. The full-recording periodogram results do not use annotation codes and have no accuracy evaluation. No motion or ROI-quality rejection is applied, so tracker and region changes may affect the estimates. Results are stored in `processed/respiration_baseline_rosbag_raw/`.

Run `python scripts/graph_denoise_signals.py` to create an exploratory sparse
graph-filter output under `processed/graph_denoising/`. The existing artifact
uses PNG ROI signals and candidate ROI coordinates to compute fixed-graph
Laplacian and iteratively reweighted GTV signals. Its temporal edges join
adjacent frame indices and do not use the recovered irregular time intervals.
Its default strengths/scales are starting values, not tuned parameters. The
report records that this adaptation omits the reference paper's learned
feature metric and trace constraint; no labels are joined and no improvement
is measured. The separate raw-count, timestamp-aware experiment is described
below.

The separate `scripts/graph_denoise_timestamped_raw.py` experiment uses raw
mono16 ROI counts, outputs both count and Celsius columns, robust subject-wide
normalization, and measured time gaps for temporal edge weights. It writes denoised signals and raw/PWS/GTV spectral
rate comparisons under `processed/graph_denoising_rosbag_raw/`. Run
`scripts/sweep_timestamped_raw_graph_parameters.py` for its 3 × 3 strength
sensitivity grid and `scripts/plot_timestamped_graph_denoising.py` for
time-axis plots. The parameters are not tuned against labels; the sweep shows
that Manuel's nose estimate ranges from 11.79 to 20.64 BPM as graph strengths
increase, and the strongest GTV setting did not converge in 15 of 45
subject/setting runs.

Run `python scripts/evaluate_graph_leave_one_subject_out.py` for exploratory
subject-wise tuning: each fold selects PWS/GTV strengths on four subjects and
reports the fifth subject as held out. Across the five held-out folds, MAE is
4.72 BPM for raw, 4.62 BPM for selected PWS, and 4.91 BPM for selected GTV.
The small PWS difference does not establish an improvement with only five
subjects and unvalidated ROIs. Results are in
`processed/graph_denoising_rosbag_raw/leave_one_subject_out_tuning.csv` and
`.json`; no final graph setting is selected.

Run `python scripts/train_tcn_inhalation_event_detector.py` for the available-
label TCN adaptation. It predicts inhalation events from nose temperature and
fixed PWS/GTV graph features, with outer leave-one-subject-out evaluation and
threshold calibration inside each training fold. Macro F1 is 0.141, below the
training-fold-selected bandpass peak/valley baseline at 0.318. The TCN is retained as an
exploratory result, not a selected model. It detects sparse events rather than
denoising a continuous waveform; the dataset does not provide a continuous
waveform target. Details are in
`processed/tcn_inhalation_event_detector/tcn_loso_report.json`.

Run `python scripts/plot_graph_denoising.py` to create five region-by-region
comparisons under `processed/graph_denoising/plots/`. These frame-index plots
are for visual inspection and do not establish respiration accuracy.

The original ROS bags are available locally under `../ros/`. Their `mono16` payloads are the source for recovered raw matrices and measured image timestamps. Derived temperatures use the documented L-CAS encoding conversion linked above; the raw matrix payloads remain unchanged.


