"""Reproduce the first-ward-entry cohort and its laboratory-only baseline."""

import argparse
import csv
import gzip
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from cxr_warning.cohort import HORIZON, ICUS, SPLITS, WARDS, read_rows, timestamp

KNOWN_UNITS = WARDS | ICUS
INDEX_FIELDS = ("hadm_id", "subject_id", "admittime", "index_time", "dischtime",
                "endpoint", "unknown_reason", "icu_event", "death_event")
ADMISSION_FIELDS = ("hadm_id", "subject_id", "admittime", "dischtime", "deathtime",
                    "hospital_expire_flag")
TRANSFER_FIELDS = ("hadm_id", "careunit", "intime", "outtime")
LAB_FIELDS = ("subject_id", "hadm_id", "itemid", "valuenum", "charttime", "storetime", "valueuom")
LAB_WINDOW = timedelta(hours=24)
SEED = 42
PANEL_LIMIT = 50
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def _time(value):
    try:
        return timestamp((value or "").strip())
    except ValueError:
        return None


def _coverage_reason(rows, index, end):
    covered = index
    for start, stop in sorted((start, stop) for unit, start, stop in rows
                              if unit in KNOWN_UNITS and stop is not None):
        if stop <= covered:
            continue
        if start > covered:
            break
        covered = stop
        if covered >= end:
            break
    if covered < end:
        return "gap"
    if any(unit not in KNOWN_UNITS and stop is not None and start < end and stop > index
           for unit, start, stop in rows):
        return "unknown_unit"
    if any(stop is None and start <= end for _, start, stop in rows):
        return "trailing_unbounded"
    return ""


def build_index(admissions, transfers):
    """Return first-ward index rows and the historical endpoint audit counts."""
    adm = {}
    for row in read_rows(admissions, ADMISSION_FIELDS):
        hadm = (row["hadm_id"] or "").strip()
        if hadm:
            adm[hadm] = ((row["subject_id"] or "").strip(), _time(row["admittime"]),
                         _time(row["dischtime"]), _time(row["deathtime"]),
                         (row["hospital_expire_flag"] or "").strip())
    stats = Counter(total_admissions=len(adm))

    first_ward = {}
    for row in read_rows(transfers, TRANSFER_FIELDS):
        hadm = (row["hadm_id"] or "").strip()
        if hadm in adm and (row["careunit"] or "").strip() in WARDS:
            start = _time(row["intime"])
            if start is not None and (hadm not in first_ward or start < first_ward[hadm]):
                first_ward[hadm] = start

    candidates = {}
    for hadm, (subject, admit, discharge, death, flag) in adm.items():
        index = first_ward.get(hadm)
        if index is None:
            stats["excluded_no_ward_entry"] += 1
        elif admit is None or discharge is None or not admit <= index < discharge:
            stats["excluded_index_outside_stay_or_bad_times"] += 1
        else:
            candidates[hadm] = (subject, admit, index, discharge, death, flag)
    del adm, first_ward
    stats["candidates"] = len(candidates)

    first_icu = {}
    for row in read_rows(transfers, TRANSFER_FIELDS):
        hadm = (row["hadm_id"] or "").strip()
        if hadm in candidates and (row["careunit"] or "").strip() in ICUS:
            start = _time(row["intime"])
            if start is not None and (hadm not in first_icu or start < first_icu[hadm]):
                first_icu[hadm] = start

    rows, pending = [], {}
    for hadm, (subject, admit, index, discharge, death, flag) in candidates.items():
        icu = first_icu.get(hadm)
        if icu is not None and icu <= index:
            stats["excluded_icu_at_or_before_index"] += 1
            continue
        end = min(index + HORIZON, discharge)
        icu_event = icu if icu is not None and icu <= end else None
        death_event = death if death is not None and index < death <= end else None
        row = dict(hadm_id=hadm, subject_id=subject, admittime=admit, index_time=index,
                   dischtime=discharge, icu_event=icu_event, death_event=death_event)
        if icu_event or death_event:
            row["endpoint"] = "positive"
        elif flag not in {"0", "1"} or (flag == "1") != (death is not None):
            row["endpoint"] = "unknown"
            stats["unknown_reason_death_flag_inconsistent"] += 1
        else:
            pending[hadm] = (index, end, row)
        rows.append(row)
    del candidates, first_icu

    intervals = defaultdict(list)
    for transfer in read_rows(transfers, TRANSFER_FIELDS):
        hadm = (transfer["hadm_id"] or "").strip()
        if hadm not in pending:
            continue
        index, end, _ = pending[hadm]
        start, stop = _time(transfer["intime"]), _time(transfer["outtime"])
        if start is not None and start <= end and (stop is None or stop > index):
            intervals[hadm].append(((transfer["careunit"] or "").strip(), start, stop))
    for hadm, (index, end, row) in pending.items():
        reason = _coverage_reason(intervals.get(hadm, ()), index, end)
        row["endpoint"] = "unknown" if reason else "negative"
        if reason:
            row["unknown_reason"] = reason
            stats[f"unknown_reason_{reason}"] += 1

    counts = Counter(row["endpoint"] for row in rows)
    stats.update(main=len(rows), main_patients=len({row["subject_id"] for row in rows}),
                 positive_union=counts["positive"], negative_covered=counts["negative"],
                 unknown_total=counts["unknown"])
    for endpoint in ("positive", "unknown"):
        stats[f"{endpoint}_patients"] = len({row["subject_id"] for row in rows
                                             if row["endpoint"] == endpoint})
    stats["check_partition"] = sum(counts[name] for name in ("positive", "negative", "unknown")) == len(rows)
    return rows, dict(stats)


def _write_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_cohort(admissions, transfers, out):
    started = time.monotonic()
    rows, stats = build_index(admissions, transfers)
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "expansion_index.csv"
    with open(artifact, "w", encoding="utf-8", newline="",
              opener=lambda path, flags: os.open(path, flags, 0o600)) as handle:
        os.fchmod(handle.fileno(), 0o600)
        writer = csv.DictWriter(handle, INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_report(out / "expansion_index_log.json", {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0], "stats": stats, "artifact": artifact.name,
        "runtime_seconds": round(time.monotonic() - started, 1),
    })


def _patient_split(subjects, labels):
    from sklearn.model_selection import train_test_split

    positive = {}
    for subject, label in zip(subjects, labels):
        positive[subject] = max(positive.get(subject, 0), label)
    patients = sorted(positive)
    train, rest = train_test_split(patients, test_size=0.30, random_state=SEED,
                                   stratify=[positive[p] for p in patients])
    val, test = train_test_split(rest, test_size=0.50, random_state=SEED,
                                 stratify=[positive[p] for p in rest])
    return {subject: name for name, group in zip(SPLITS, (train, val, test)) for subject in group}


def _visible_labs(path, by_patient, stats, pass_name, panel=None, included=None):
    opener = gzip.open if path.suffix == ".gz" else open
    rank_pass = panel is None
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        missing = set(LAB_FIELDS) - set(header)
        if missing:
            raise ValueError(f"{path.name} missing columns: {', '.join(sorted(missing))}")
        sub, hadm, item, num, chart_col, store_col, unit = map(header.index, LAB_FIELDS)
        rows_read = f"{pass_name}_rows_read"
        time_rejected = f"{pass_name}_time_rejected"
        for row in reader:
            stats[rows_read] += 1
            candidates = by_patient.get(row[sub])
            if not candidates:
                if rank_pass:
                    stats["pass1_skipped_no_candidate"] += 1
                continue
            itemid, raw = row[item].strip(), row[num].strip()
            if not itemid or not raw:
                if rank_pass:
                    stats["pass1_skipped_no_item_or_value"] += 1
                continue
            try:
                value = float(raw)
            except ValueError:
                if rank_pass:
                    stats["pass1_skipped_bad_value"] += 1
                continue
            if not math.isfinite(value):
                if rank_pass:
                    stats["pass1_skipped_nonfinite"] += 1
                continue
            key = (itemid, row[unit].strip())
            if not rank_pass and key not in panel:
                continue
            chart, store = row[chart_col].strip(), row[store_col].strip()
            if not chart or not store:
                if rank_pass:
                    stats["pass1_skipped_bad_time"] += 1
                continue
            lab_hadm = row[hadm].strip()
            for i, admission, admit, lo, hi, index in candidates:
                if included is not None and i not in included:
                    continue
                if lab_hadm != admission and (lab_hadm or not chart < admit):
                    continue
                # The historical scans compare canonical MIMIC timestamp strings.
                if not (lo <= chart <= hi and chart <= store <= index):
                    stats[time_rejected] += 1
                    continue
                yield i, key, value, chart, store


def _lab_matrix(labevents, by_patient, row_split, stats):
    import numpy as np

    included = set()
    counts = Counter()
    for i, key, _, _, _ in _visible_labs(labevents, by_patient, stats, "pass1"):
        included.add(i)
        if row_split[i] == "train":
            counts[key] += 1
    cohort = sorted(included)
    positions = {i: j for j, i in enumerate(cohort)}
    # Rank matched records before deduplication, including repeated draws.
    panel = sorted(counts, key=lambda key: (-counts[key], key))[:PANEL_LIMIT]
    columns = {key: j for j, key in enumerate(panel)}
    shape = (len(cohort), len(panel))
    values = np.full(shape, np.nan, dtype=np.float32)
    charts = np.full(shape, np.inf)
    stores = np.full(shape, np.inf)
    for i, key, value, chart, store in _visible_labs(labevents, by_patient, stats, "pass2",
                                                     columns, positions):
        row, col = positions[i], columns[key]
        stamp = (datetime.fromisoformat(chart).timestamp(), datetime.fromisoformat(store).timestamp())
        if stamp < (charts[row, col], stores[row, col]):
            charts[row, col], stores[row, col] = stamp
            values[row, col] = value
    return cohort, panel, values


def _split_summary(indices, subjects, labels, row_split):
    report = {}
    for name in SPLITS:
        members = [i for i in indices if row_split[i] == name]
        positives = [i for i in members if labels[i]]
        report[name] = {
            "admissions": len(members), "patients": len({subjects[i] for i in members}),
            "positive_admissions": len(positives), "positive_patients": len({subjects[i] for i in positives}),
            "positive_rate": round(len(positives) / len(members), 6) if members else 0.0,
        }
    return report


def _specimen_mix(path, panel):
    itemids = {itemid for itemid, _ in panel}
    lookup = {}
    for row in read_rows(path, ("itemid", "fluid", "category")):
        itemid = (row["itemid"] or "").strip()
        if itemid in itemids:
            lookup[itemid] = ((row["fluid"] or "").strip(), (row["category"] or "").strip())
    specimens = [lookup.get(itemid, ("unknown", "unknown")) for itemid, _ in panel]
    fluids = Counter(fluid for fluid, _ in specimens)
    return {
        "panel_size": len(panel), "fluid_counts": dict(fluids),
        "category_counts": dict(Counter(category for _, category in specimens)),
        "blood_fluid_items": fluids["Blood"], "non_blood_items": len(panel) - fluids["Blood"],
    }


def _fit(values, labels, row_split):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    masks = {name: row_split == name for name in SPLITS}
    if any(len(np.unique(labels[mask])) != 2 for mask in masks.values()):
        raise ValueError("Each split needs both endpoints after filtering for visible labs")
    train = masks["train"]
    raw_train = values[train]
    mean, std = np.zeros(values.shape[1]), np.ones(values.shape[1])
    for j, column in enumerate(raw_train.T):
        observed = column[np.isfinite(column)].astype(np.float64)
        if observed.size:
            mean[j] = observed.mean()
            sd = observed.std()
            std[j] = sd if sd > 1e-8 else 1.0
    finite = np.isfinite(values)
    z = np.where(finite, (values.astype(np.float64) - mean) / std, 0.0).astype(np.float32)
    features = np.hstack([z, finite])
    del z, finite, raw_train
    x_train, y_train = features[train], labels[train]
    x_val, y_val = features[masks["val"]], labels[masks["val"]]

    def fit(c):
        return LogisticRegression(C=c, solver="lbfgs", max_iter=2000,
                                  random_state=SEED).fit(x_train, y_train)

    grid = {str(c): float(average_precision_score(y_val, fit(c).predict_proba(x_val)[:, 1]))
            for c in C_GRID}
    chosen = max(C_GRID, key=lambda c: grid[str(c)])
    final = fit(chosen)
    metrics = {}
    for name, mask in masks.items():
        y = labels[mask]
        p = final.predict_proba(features[mask])[:, 1]
        metrics[f"{name}_metrics"] = {"auprc": float(average_precision_score(y, p)),
                                      "auroc": float(roc_auc_score(y, p)),
                                      "brier": float(np.mean((p - y) ** 2))}
    return chosen, grid, metrics


def train_baseline(index, labevents, d_labitems=None):
    """Return aggregate training and verification reports without exporting a model."""
    import numpy as np
    import sklearn

    started = time.monotonic()
    subjects, labels = [], []
    by_patient = defaultdict(list)
    for row in read_rows(index, INDEX_FIELDS[:4] + ("endpoint",)):
        if row["endpoint"] not in ("positive", "negative"):
            continue
        admit = datetime.fromisoformat(row["admittime"])
        lo, hi = ((admit + shift).isoformat(sep=" ") for shift in (-LAB_WINDOW, LAB_WINDOW))
        subject = row["subject_id"]
        by_patient[subject].append((len(subjects), row["hadm_id"], row["admittime"], lo, hi, row["index_time"]))
        subjects.append(subject)
        labels.append(int(row["endpoint"] == "positive"))
    assignment = _patient_split(subjects, labels)
    row_split = [assignment[subject] for subject in subjects]
    del assignment
    stats = Counter(decidable=len(subjects), patient_zero_overlap=True)
    cohort, panel, values = _lab_matrix(labevents, by_patient, row_split, stats)
    del by_patient
    stats.update(cohort_total=len(cohort), cohort_positive=sum(labels[i] for i in cohort), panel_size=len(panel))
    cohort_patients = len({subjects[i] for i in cohort})
    pre_filter = _split_summary(range(len(subjects)), subjects, labels, row_split)
    for summary in pre_filter.values():
        summary["positives"] = summary.pop("positive_admissions")
        summary.pop("positive_patients")
    post_filter = _split_summary(cohort, subjects, labels, row_split)
    verification = {
        "scope": "aggregates_only",
        "denominators": {"decidable": len(subjects), "cohort_total": len(cohort),
                         "cohort_positive": stats["cohort_positive"], "cohort_patients": cohort_patients},
        "in_cohort_split_stats": post_filter,
        "assertions": {
            "split_sums_match_cohort": (sum(s["admissions"] for s in post_filter.values()) == len(cohort)
                                        and sum(s["positive_admissions"] for s in post_filter.values())
                                        == stats["cohort_positive"]),
            "patient_zero_cross_split": sum(s["patients"] for s in post_filter.values()) == cohort_patients,
        },
    }
    if d_labitems is not None:
        verification["panel_specimen_check"] = _specimen_mix(d_labitems, panel)
    chosen, grid, metrics = _fit(values, np.array([labels[i] for i in cohort], dtype=np.int8),
                                 np.array([row_split[i] for i in cohort]))
    report = {
        "status": "completed", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "sklearn": sklearn.__version__, "seed": SEED,
        "model": "L2 logistic regression (lbfgs); train-mean imputation and observation indicators",
        "panel_selection": "top 50 (itemid, unit) by non-deduplicated training record frequency",
        "c_grid": list(C_GRID), "chosen_c": chosen, "grid_val_auprc": grid, "splits": pre_filter,
        "cohort": {"decidable": len(subjects), "included_total": len(cohort),
                   "included_positive": stats["cohort_positive"]},
        **metrics, "panel": [{"itemid": itemid, "unit": unit} for itemid, unit in panel],
        "stats": dict(stats), "runtime_seconds": round(time.monotonic() - started, 1),
    }
    return report, verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cohort = commands.add_parser("cohort", help="build the first-ward-entry index")
    train = commands.add_parser("train", help="train and report the laboratory-only baseline")
    for command, flags in ((cohort, ("admissions", "transfers", "out")),
                           (train, ("index", "labevents", "out"))):
        for flag in flags:
            command.add_argument(f"--{flag}", type=Path, required=True)
    train.add_argument("--d-labitems", type=Path, help="optional lab item dictionary for specimen counts")
    args = parser.parse_args(argv)
    if args.command == "cohort":
        run_cohort(args.admissions, args.transfers, args.out)
    else:
        report, verification = train_baseline(args.index, args.labevents, args.d_labitems)
        args.out.mkdir(parents=True, exist_ok=True)
        _write_report(args.out / "labs_baseline_log.json", report)
        _write_report(args.out / "verify_split_panel_log.json", verification)


if __name__ == "__main__":
    main()
