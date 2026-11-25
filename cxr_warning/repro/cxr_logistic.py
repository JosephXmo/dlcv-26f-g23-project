import argparse
from itertools import takewhile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ..__main__ import report_json
from ..cohort import read_rows

CS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def prefix(archive, name, count, tail=()):
    with archive.open(f"{name}.npy") as handle:
        version = np.lib.format.read_magic(handle)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError(f"{name}: unsupported NPY header")
        shape, fortran, dtype = readers[version](handle)
        valid_dtype = dtype.kind == "f" and dtype.itemsize in (4, 8) if tail else (
            dtype.kind == "U" and 0 < dtype.itemsize <= 128
        )
        if not shape or shape[0] < count or shape[1:] != tail or fortran or not valid_dtype:
            raise ValueError(f"{name}: invalid cache shape, layout or dtype")
        size = count * (tail[0] if tail else 1) * dtype.itemsize
        return np.frombuffer(handle.read(size), dtype=dtype).reshape((count, *tail))


def load_train_val(manifest, features):
    rows = list(takewhile(lambda row: row["split"] != "test",
                          read_rows(manifest, ("hadm_id", "subject_id", "split", "risk"))))
    splits = [row["split"] for row in rows]
    n_train = splits.count("train")
    if not 0 < n_train < len(rows) or splits != ["train"] * n_train + ["val"] * (len(rows) - n_train):
        raise ValueError("Expected contiguous train and validation rows before any test rows")
    ids = [row["hadm_id"] for row in rows]
    patients = np.array([row["subject_id"] for row in rows])
    risk = np.array([int(row["risk"]) for row in rows])
    if ("" in ids or len(set(ids)) != len(ids) or "" in patients or
            set(patients[:n_train]) & set(patients[n_train:])):
        raise ValueError("Invalid admission identifiers or patients shared between train and validation")
    if set(risk[:n_train]) != {0, 1} or set(risk[n_train:]) != {0, 1}:
        raise ValueError("Train and validation must both contain cases and controls")
    with ZipFile(features) as archive:
        if prefix(archive, "hadm_id", len(rows)).tolist() != ids:
            raise ValueError("Cache does not match the train and validation admission order")
        cxr, ecg, labs = (prefix(archive, name, len(rows), (width,)) for name, width in
                          (("cxr", 1024), ("ecg", 84), ("labs", 50)))
    if not np.isfinite(cxr).all() or not np.isfinite(ecg).all() or np.isinf(labs).any():
        raise ValueError("Expected finite CXR and ECG features, with NaNs allowed only in labs")
    return np.column_stack((cxr, ecg, labs)).astype(np.float64), np.isfinite(labs), risk, patients, n_train


def fit(manifest, features):
    x, observed, y, groups, n_train = load_train_val(manifest, features)
    train = np.arange(n_train)
    val = np.arange(n_train, len(y))

    def prep(a, b):
        scaler = StandardScaler().fit(x[a])
        return [np.column_stack((np.nan_to_num(scaler.transform(x[i])), observed[i])) for i in (a, b)]

    def scores(c, a, ya, b):
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=2000).fit(a, ya)
        if model.n_iter_[0] >= model.max_iter:
            raise ValueError("Logistic regression did not converge")
        return model.predict_proba(b)[:, 1]

    cv = {c: [] for c in CS}
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for a, b in folds.split(x[train], y[train], groups[train]):
        if set(y[a]) != {0, 1} or set(y[b]) != {0, 1}:
            raise ValueError("Every cross-validation fold must contain cases and controls")
        xa, xb = prep(a, b)
        for c in CS:
            cv[c].append(average_precision_score(y[b], scores(c, xa, y[a], xb)))
    cv = {c: float(np.mean(values)) for c, values in cv.items()}
    selected = max(CS, key=cv.get)
    xa, xb = prep(train, val)
    pred = scores(selected, xa, y[train], xb)
    return {"sklearn_version": sklearn_version, "selected_C": selected, "train_5fold_group_mean_auprc": cv,
            "validation": {"auprc": float(average_precision_score(y[val], pred)),
                           "auroc": float(roc_auc_score(y[val], pred)),
                           "brier": float(brier_score_loss(y[val], pred))}}


def main():
    parser = argparse.ArgumentParser(description="Fit the report's validation-only CXR + ECG + labs L2 baseline")
    for name in ("manifest", "features", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report_json(args.out, fit(args.manifest, args.features))


if __name__ == "__main__":
    main()
