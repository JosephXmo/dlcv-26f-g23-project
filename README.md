# Chest X-ray–centered 48-hour ward risk prediction

Course project prototype for predicting first ICU transfer or in-hospital death from chest X-ray, ECG and laboratory results. It compares image-only and clinical-only models against concatenation and gated fusion, with separate modality-dropout and CheXpert auxiliary-task ablations.

## Quick start

```sh
uv sync
uv run python -m cxr_warning smoke --out runs/demo
```

The smoke run generates timestamped synthetic admissions, retrieval candidates, transfers and lab records. Synthetic feature vectors stand in for CXR and ECG. `runs/demo/cohort_audit.json` contains the cohort flow and event lead times. `runs/demo/metrics.json` contains all six experiments and identifies their source as `synthetic`. If PyTorch crashes in CPU backward on WSL2, try disabling oneDNN for that run: `uv run python -c 'import runpy, torch; torch.backends.mkldnn.enabled = False; runpy.run_module("cxr_warning", run_name="__main__")' smoke --out runs/demo`.

## Reproduce the course report

The following commands correspond to [the course report](REPORT.md). The synthetic smoke run above and the prototype workflow below are not reproductions of the reported results. Follow the PhysioNet access agreements; use local copies of the [Symile-MIMIC 1.0.0](https://physionet.org/content/symile-mimic/1.0.0/), [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/3.1/), and [MIMIC-IV-ECG 1.0](https://physionet.org/content/mimic-iv-ecg/1.0/) files. The report workflow does not use original CXR JPG files.

Use a private output location on a filesystem that enforces permissions. Git ignores repository-root `data/`, `runs/`, and `outputs/`, but Git ignore rules do not control filesystem access. Never commit input data, per-admission tables, features, predictions, or weights.

```sh
uv sync --locked --extra repro
umask 077
MIMIC=data/mimiciv-3.1/hosp
SYMILE=data/symile-mimic-1.0.0
ECG=data/mimic-iv-ecg-1.0
OUT="$HOME/cxr-report-local"
mkdir -p "$OUT"
chmod 700 "$OUT"
```

### Primary experiment: CXR-indexed cohort and six neural models

`$SYMILE` must contain `symile_mimic_data.csv`, `train.csv`, `val.csv`, `test.csv` and the published `data_npy/{train,val,test}/` with `cxr_<split>.npy` and `hadm_id_<split>.npy`; the test directory also needs `label_test.npy` and `label_hadm_id_test.npy`. As in the historical cohort script, MIMIC-IV 3.1 admissions establish membership and transfers determine ward and ICU timing; Symile metadata supplies admission, discharge, and death fields.

Place the [official torchvision DenseNet121 ImageNet weights](https://download.pytorch.org/models/densenet121-a639ec97.pth) in the `checkpoints` subdirectory of `torch.hub.get_dir()` (typically `~/.cache/torch/hub`). Extraction checks the checksum prefix and runs offline. It center-crops the published tensors without renormalizing. `$ECG` points to unpacked WFDB files at their metadata-relative paths. The historical run separately cross-checked ECG patient, path, and time against the official `record_list.csv`; this runner does not repeat that cross-check, so verify it locally before using the recipe for an exact replication.

```sh
mkdir -p "$OUT/cxr"
uv run --locked --extra repro python -m cxr_warning.repro.cxr cohort \
  --symile "$SYMILE" --admissions "$MIMIC/admissions.csv.gz" \
  --transfers "$MIMIC/transfers.csv.gz" --out "$OUT/cxr/cohort.csv" \
  > "$OUT/cxr/cohort_audit.json"
uv run --locked --extra repro python -m cxr_warning.repro.cxr extract \
  --manifest "$OUT/cxr/cohort.csv" --data-npy "$SYMILE/data_npy" \
  --ecg-root "$ECG" --labevents "$MIMIC/labevents.csv.gz" \
  --out "$OUT/cxr/features.npz"
uv run --locked --extra repro python -m cxr_warning train \
  --manifest "$OUT/cxr/cohort.csv" --features "$OUT/cxr/features.npz" \
  --epochs 25 --seed 42 --bootstrap 500 --specificity 0.90 \
  --dropout 0.25 --aux-weight 0.25 --out "$OUT/cxr"
```

Check `cohort_audit.json` against the final train/validation/test counts (3,950 / 310 / 198, including 324 / 22 / 26 positives). `metrics.json` reports the six models, AP (`auprc`), paired intervals, AP/PA results, and the simulated missing-modality scenarios.

### Primary experiment: validation-only all-modality logistic baseline

This separate command uses only the contiguous train/validation prefix of the primary feature cache. It selects `C` by five-fold patient-grouped cross-validation within the training set and scores only validation. It does not score the CXR test set.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --locked --extra repro python -m cxr_warning.repro.cxr_logistic \
  --manifest "$OUT/cxr/cohort.csv" --features "$OUT/cxr/features.npz" \
  --out "$OUT/cxr/logreg_trainval.json"
```

### Secondary experiment: first-ward-entry lab baseline

This cohort starts at first ordinary-ward entry, not at CXR acquisition. Run both stages independently of the primary experiment; no imaging or ECG files are needed.

```sh
uv run --locked --extra repro python -m cxr_warning.repro.ward_labs cohort \
  --admissions "$MIMIC/admissions.csv.gz" --transfers "$MIMIC/transfers.csv.gz" \
  --out "$OUT/ward-entry"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --locked --extra repro python -m cxr_warning.repro.ward_labs train \
  --index "$OUT/ward-entry/expansion_index.csv" --labevents "$MIMIC/labevents.csv.gz" \
  --d-labitems "$MIMIC/d_labitems.csv.gz" --out "$OUT/ward-entry"
```

`labs_baseline_log.json` contains the metrics. Compare `verify_split_panel_log.json` with the report: 241,018 admissions and 10,456 positives after lab filtering, the per-split counts, and 46 blood / 4 urine panel entries. The historical logistic runs used scikit-learn 1.3.0; the lockfile uses a newer build compatible with the repository's NumPy stack. Treat the reported real-data figures as comparison targets, not as results verified with this lockfile. The reproduction unit tests use only synthetic fixtures: `uv run --locked --extra repro python -m unittest discover -s tests`.

## Original prototype workflow (not the report)

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
