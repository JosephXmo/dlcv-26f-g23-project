# Chest X-ray–centered 48-hour ward risk prediction

Course project prototype for predicting first ICU transfer or in-hospital death from chest X-ray, ECG and laboratory results. It compares image-only and clinical-only models against concatenation and gated fusion, with separate modality-dropout and CheXpert auxiliary-task ablations.

## Quick start

```sh
uv sync
uv run python -m cxr_warning smoke --out runs/demo
```

The smoke run generates timestamped synthetic admissions, retrieval candidates, transfers and lab records. Synthetic feature vectors stand in for CXR and ECG. `runs/demo/cohort_audit.json` contains the cohort flow and event lead times. `runs/demo/metrics.json` contains all six experiments and identifies their source as `synthetic`.

## Run with Symile-MIMIC

Use local copies of the credentialed files from [Symile-MIMIC 1.0.0](https://physionet.org/content/symile-mimic/1.0.0/), [MIMIC-IV 2.2](https://physionet.org/content/mimiciv/2.2/), [MIMIC-CXR-JPG 2.0.0](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and [MIMIC-IV-ECG 1.0](https://physionet.org/content/mimic-iv-ecg/1.0/). Keep the datasets and row-level outputs under ignored directories such as `data/` and `runs/`.

Run the cohort audit first. The Symile directory must contain `symile_mimic_data.csv`, `train.csv`, `val.csv` and `test.csv`. The paths below are examples. `transfers` and `labevents` accept CSV or gzipped CSV.

```sh
uv run python -m cxr_warning cohort \
  --symile data/symile \
  --transfers data/mimiciv/hosp/transfers.csv.gz \
  --out runs/mimic
```

Check exclusions and event counts by split in `cohort_audit.json`. Confirm the ward allowlist against `index_locations`. Additional confirmed ward names can be supplied with `--ward 'Unit Name'`. Review transfers with short lead times and any evidence of planned postoperative ICU admissions before interpreting results.

```sh
uv sync --extra mimic
uv run --extra mimic python -m cxr_warning extract \
  --manifest runs/mimic/cohort.csv \
  --cxr-root data/mimic-cxr-jpg \
  --ecg-root data/mimic-iv-ecg \
  --labevents data/mimiciv/hosp/labevents.csv.gz \
  --out runs/mimic/features.npz
uv run python -m cxr_warning train \
  --manifest runs/mimic/cohort.csv \
  --features runs/mimic/features.npz \
  --out runs/mimic
```

The imaging roots resolve the relative `cxr_path` and `ecg_path` in the manifest. The extractor downloads the CheXpert-trained DenseNet-121 from TorchXRayVision on first use and caches a 1,024-dimensional embedding per CXR. ECG uses per-lead waveform statistics. The blood panel uses up to 50 `(itemid, unit)` pairs ranked by training-cohort frequency, with per-feature observation masks. Extraction accepts `--device mps` or `--device cuda` when available. Fusion training runs on CPU.

## Cohort and evaluation rules

- The index is the Symile frontal CXR timestamp. The patient must be alive and unambiguously on a ward, with no ICU transfer through that time in the current admission. The first subsequent ICU transfer or recorded in-hospital death is positive if it falls in `(index, index + 48h]` and occurs by discharge. Surviving discharge without an event is negative. The prototype includes planned transfers when the transfer data do not distinguish them.
- Use only the matched candidate for each test query to restore one row per admission, and verify patient-level split separation. ECG uses acquisition time as an availability proxy. Labs use numeric results from the 24-hour window on either side of admission, with `charttime ≤ storetime ≤ index`.
- Both risk and auxiliary heads can update the trainable CXR projection. The auxiliary head uses CXR features and binary cross-entropy on known 0/1 report-derived CheXpert labels. Blank and uncertain labels are masked. Concat and gate receive identical modality-availability inputs and dropout rates in the primary fusion comparison.
- Select checkpoints by validation AUPRC and set an operating threshold on validation at a target of 90% specificity. Report test AUPRC (average precision), AUROC, Brier score, calibration bins, sensitivity/specificity and AP/PA breakdowns. Compare AUPRC differences using paired patient-level bootstrap. On the complete-modality test subset, evaluate four scenarios from all modalities available to CXR only.
