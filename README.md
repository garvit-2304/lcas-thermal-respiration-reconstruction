# L-CAS Thermal Respiration Reconstruction

A reproducible research pipeline for recovering timestamped thermal signals from the L-CAS recordings and evaluating respiration-rate and inhalation-event baselines.

## Repository contents

- `scripts/` contains the project source code for frame and bag audits, timestamp synchronization, ROI extraction, signal filtering, rate/event evaluation, experiments, and report generation.
- `docs/workflow.md` documents the project methods, experiments, current findings, and limitations.
- `docs/data_setup.md` and `docs/annotation_reference.md` describe the expected local data layout and annotation workflow.
- `reports/thermal_respiration_project_report.pdf` is this reconstruction's results report, kept separate from the CNN/GRU-MRF progress report for the other attempt.

## Data

The source recordings and generated participant-level data are not included. Obtain the L-CAS dataset under its applicable terms and place the files in the local paths expected by the scripts (`original_dataset/` for source frames and physiology CSVs, `ros/` for the original ROS bags). Generated outputs under `data/` and temporary files are ignored by Git.

Do not commit source thermal images, ROS bags, participant-level CSVs, or standalone face-preview images to this repository. The private repository includes the user-requested report PDF, which contains selected derived result figures.

## Environment

Use Python 3.10-3.12. Install dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The source includes dataset audits, timestamp recovery, raw thermal matrix extraction, geometric face/ROI tracking, raw-count and Celsius signal extraction, Butterworth/periodogram baselines, graph filtering, leave-one-subject-out evaluation, and a sparse inhalation-event TCN experiment. Individual scripts document command-line parameters via `--help` where available. The detailed workflow is in `docs/workflow.md`.

## Evidence and limitations

The project processes five recordings and 14,154 thermal frames using measured image-header timestamps. Results remain exploratory: there are only five subjects, the numeric respiration-code map is owner-confirmed rather than author-verified, and the facial-region reference review covers a small single-reviewer sample. The tested TCN did not beat the bandpass event comparator; graph filtering did not show a convincing improvement over raw nose signals. See the report and workflow document for metrics and caveats.

## License

No license is included. Dataset use remains subject to the original L-CAS terms.

