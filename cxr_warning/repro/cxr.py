"""Rebuild the historical 48-hour ward cohort and its local multimodal feature cache."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import numpy as np

from ..cohort import HORIZON, ICUS, MANIFEST, SPLITS, WARDS, read_rows, timestamp, write_rows
from ..features import LAB_COLUMNS, LEADS, ecg_features, save_features

KNOWN = WARDS | ICUS
LAB_WINDOW = timedelta(hours=24)
SOURCE = "real:symile_cxr_imagenet_densenet121+mimic_iv_ecg+mimic_iv_3.1_labs"


def _time(value):
    try:
        parsed = timestamp((value or "").strip())
        return parsed if parsed is None or parsed.tzinfo is None else None
    except ValueError:
        return None


def _membership(symile):
    membership, patients = {}, {}
    for split in SPLITS:
        columns = ("hadm_id", "subject_id")
        if split == "test":
            columns += ("label_hadm_id", "label")
        positives = defaultdict(list)
        candidates = []
        for row in read_rows(symile / f"{split}.csv", columns):
            hadm, subject = ((row[k] or "").strip() for k in columns[:2])
            if split != "test":
                candidates.append((hadm, subject))
            elif hadm and (row["label_hadm_id"] or "").strip() == hadm and (row["label"] or "").strip() == "1":
                positives[hadm].append((hadm, subject))
        if split == "test":
            candidates = [matches[0] for matches in positives.values() if len(matches) == 1]
        for hadm, subject in candidates:
            if hadm and subject and hadm not in membership and patients.get(subject, split) == split:
                membership[hadm] = split, subject
                patients[subject] = split
    return membership


def _coverage_gap(stays, index, end):
    intervals = sorted((start, stop) for unit, start, stop, _ in stays
                       if unit in KNOWN and start and stop and start < stop)
    cursor = index
    for start, stop in intervals:
        if stop <= cursor:
            continue
        if start > cursor:
            break
        cursor = stop
        if cursor >= end:
            break
    if cursor < end:
        return "gap_in_valid_known_unit_intervals"
    if any(unit not in KNOWN and start and stop and start < end and stop > index
           for unit, start, stop, _ in stays):
        return "unknown_unit_in_window"
    if any(missing_stop and start and start <= end for _, start, _, missing_stop in stays):
        return "trailing_unbounded_transfer_in_window"
    return None


def _cohort_row(row, stays):
    admit, discharge, index = (_time(row[k]) for k in ("admittime", "dischtime", "cxr_StudyDateTime"))
    if not (admit and discharge and index and admit <= index < discharge):
        return "excluded_cxr_outside_stay", None
    if not (row["cxr_path"] or "").strip() or (row["cxr_ViewPosition"] or "").strip() not in {"AP", "PA"}:
        return "excluded_missing_frontal_cxr", None
    flag = (row["hospital_expire_flag"] or "").strip()
    death_text = (row["deathtime"] or "").strip()
    death = _time(death_text)
    if (flag not in {"0", "1"} or (flag == "1") != bool(death) or (death_text and not death)
            or (death and not index < death <= discharge)):
        return "excluded_inconsistent_death_or_dead_at_index", None
    icu_starts = [start for unit, start, _, _ in stays if unit in ICUS]
    if any(start and start <= index for start in icu_starts):
        return "excluded_icu_at_or_before_index", None
    active = [unit for unit, start, stop, _ in stays if start and stop and start <= index < stop]
    if len(active) != 1 or active[0] not in WARDS:
        return "excluded_no_unambiguous_ward_at_index", None
    if None in icu_starts:
        return "excluded_icu_missing_intime", None

    end = min(index + HORIZON, discharge)
    icu = min((start for start in icu_starts if start <= end), default=None)
    death = death if death and death <= end else None
    first = min((event for event in (icu, death) if event), default=None)
    if not first and (gap := _coverage_gap(stays, index, end)):
        return f"unknown_{gap}", None
    ecg_time = _time(row.get("ecg_time"))
    ecg_path = (row.get("ecg_path") or "").strip() if ecg_time and ecg_time <= index else ""
    return "positive" if first else "covered_negative", dict(
        row, ecg_path=ecg_path, index_ward=active[0], risk=int(first is not None),
        icu_48h=int(icu is not None), death_48h=int(death is not None),
        lead_hours=(first - index).total_seconds() / 3600 if first else None,
    )


def build_cohort(symile, admissions, transfers):
    membership = _membership(symile)
    metadata = {}
    for row in read_rows(symile / "symile_mimic_data.csv", (
        "hadm_id", "admittime", "dischtime", "cxr_StudyDateTime", "cxr_ViewPosition", "cxr_path",
        "hospital_expire_flag", "deathtime",
    )):
        hadm = (row["hadm_id"] or "").strip()
        if hadm in membership:
            metadata.setdefault(hadm, row)
    admitted = {hadm for row in read_rows(admissions, ("hadm_id",))
                if (hadm := (row["hadm_id"] or "").strip()) in membership}
    episodes = defaultdict(list)
    for row in read_rows(transfers, ("hadm_id", "subject_id", "careunit", "intime", "outtime")):
        hadm, subject = ((row[k] or "").strip() for k in ("hadm_id", "subject_id"))
        if hadm in membership and subject == membership[hadm][1]:
            episodes[hadm].append(((row["careunit"] or "").strip(), _time(row["intime"]),
                                   _time(row["outtime"]), not (row["outtime"] or "").strip()))

    manifest = []
    audit = {split: Counter() for split in SPLITS}
    for hadm, (split, subject) in membership.items():
        audit[split]["input"] += 1
        if hadm not in admitted:
            status = "excluded_mimic_admissions_unmatched"
        elif hadm not in episodes:
            status = "excluded_mimic_transfers_unmatched"
        elif hadm not in metadata:
            raise ValueError("A matched admission is missing from Symile metadata")
        else:
            status, row = _cohort_row(metadata[hadm], episodes[hadm])
            if row is not None:
                manifest.append(dict(row, hadm_id=hadm, subject_id=subject, split=split))
        audit[split][status] += 1
    return manifest, audit


def _local_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Expected a regular file within the input root")
    return path


def _cxr_locations(rows, root):
    locations, arrays = {}, {}
    for split in SPLITS:
        wanted = {row["hadm_id"] for row in rows if row["split"] == split}
        if not wanted:
            continue

        def load(name):
            return np.load(_local_file(root, f"{split}/{name}_{split}.npy"),
                           mmap_mode="r", allow_pickle=False)

        ids = load("hadm_id")
        if ids.ndim != 1:
            raise ValueError(f"Invalid CXR admission array in {split}")
        if split == "test":
            labels, queries = load("label"), load("label_hadm_id")
            if labels.shape != ids.shape or queries.shape != ids.shape:
                raise ValueError("Mismatched CXR test retrieval arrays")
            candidates = ((i, hadm) for i, (hadm, label, query) in enumerate(zip(ids, labels, queries))
                          if int(label) == 1 and int(query) == int(hadm))
        else:
            candidates = enumerate(ids)
        for i, value in candidates:
            hadm = str(int(value))
            if hadm in wanted:
                locations[hadm] = split, i
                wanted.remove(hadm)
        if wanted:
            raise ValueError(f"Unmapped cohort rows in {split}: {len(wanted)}")
        array = load("cxr")
        if array.shape != (len(ids), 3, 320, 320) or array.dtype.kind != "f":
            raise ValueError(f"Invalid CXR tensor array in {split}")
        arrays[split] = array
    return [locations[row["hadm_id"]] for row in rows], arrays


def _backbone():
    import torch
    from torchvision.models import DenseNet121_Weights, densenet121

    # Use the torchvision cache explicitly: extracting local data must not start a download.
    filename = DenseNet121_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[-1]
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not checkpoint.is_file():
        raise ValueError(f"Place the official {filename} weights in the local torch hub checkpoints directory")
    hash_prefix = checkpoint.stem.rsplit("-", 1)[-1]
    with checkpoint.open("rb") as handle:
        if not hashlib.file_digest(handle, "sha256").hexdigest().startswith(hash_prefix):
            raise ValueError(f"Checksum mismatch for {filename}")
        handle.seek(0)
        state = torch.load(handle, map_location="cpu", weights_only=True)
    # The original ImageNet checkpoint predates the current DenseNet parameter names.
    pattern = re.compile(r"(denselayer\d+\.(?:norm|relu|conv))\.([12]\.(?:weight|bias|running_mean|running_var))$")
    model = densenet121(weights=None).cpu().eval()
    model.load_state_dict({pattern.sub(r"\1\2", key): value for key, value in state.items()})
    return model


def _cxr_features(rows, root, batch_size):
    import torch
    from torch.nn.functional import adaptive_avg_pool2d
    from torchvision.transforms import CenterCrop

    locations, arrays = _cxr_locations(rows, root.resolve())
    torch.set_num_threads(4)
    model, crop = _backbone(), CenterCrop(224)
    embeddings = np.empty((len(rows), 1024), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = locations[start:start + batch_size]
            tensor = torch.empty((len(chunk), 3, 224, 224), dtype=torch.float32)
            groups = defaultdict(list)
            for pos, (split, idx) in enumerate(chunk):
                groups[split].append((pos, idx))
            for split, members in groups.items():
                positions, indices = zip(*members)
                raw = arrays[split][list(indices)]
                tensor[list(positions)] = crop(torch.from_numpy(np.ascontiguousarray(raw)).float())
            embeddings[start:start + len(chunk)] = adaptive_avg_pool2d(model.features(tensor), 1).flatten(1).numpy()
    if not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite CXR embeddings")
    return embeddings


def _lab_features(rows, path):
    by_patient = defaultdict(list)
    for idx, row in enumerate(rows):
        admitted, index = _time(row["admittime"]), _time(row["cxr_StudyDateTime"])
        if admitted and index:
            by_patient[row["subject_id"]].append((idx, row["hadm_id"], admitted, index))

    selected = [{} for _ in rows]
    for lab in read_rows(path, LAB_COLUMNS):
        candidates = by_patient.get(lab["subject_id"])
        if not candidates or not lab["valuenum"] or not lab["itemid"]:
            continue
        chart, store = _time(lab["charttime"]), _time(lab["storetime"])
        if not chart or not store:
            continue
        try:
            value = float(lab["valuenum"])
        except ValueError:
            continue
        if not np.isfinite(value):
            continue
        key = lab["itemid"], (lab["valueuom"] or "").strip()
        for idx, hadm, admitted, index in candidates:
            if lab["hadm_id"] != hadm and (lab["hadm_id"] or not chart < admitted):
                continue
            if not (admitted - LAB_WINDOW <= chart <= admitted + LAB_WINDOW and chart <= store <= index):
                continue
            previous = selected[idx].get(key)
            if previous is None or (chart, store) < previous[:2]:
                selected[idx][key] = (chart, store, value)

    frequencies = Counter(key for row, values in zip(rows, selected) if row["split"] == "train" for key in values)
    panels = sorted(frequencies, key=lambda key: (-frequencies[key], key))[:50]
    if not panels:
        raise ValueError("No eligible train lab panels")
    labs = np.full((len(rows), len(panels)), np.nan, dtype=np.float32)
    for idx, values in enumerate(selected):
        for col, panel in enumerate(panels):
            if panel in values:
                value = values[panel][2]
                if abs(value) > np.finfo(np.float32).max:
                    raise ValueError("Lab value exceeds the feature dtype range")
                labs[idx, col] = value
    return labs, panels


def _bounded_read(path, limit):
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("ECG file exceeds size limit")
    return data


def _ecg_pair(root, raw_path):
    rel = PurePosixPath(raw_path)
    if (rel.is_absolute() or not rel.parts
            or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part == ".." for part in rel.parts)):
        raise ValueError("Invalid ECG record path")
    if rel.suffix in {".hea", ".dat"}:
        rel = rel.with_suffix("")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", rel.name):
        raise ValueError("Invalid ECG record name")
    header = _bounded_read(_local_file(root, rel.with_suffix(".hea")), 65536)
    lines = [line.split() for line in header.splitlines() if line.strip() and not line.lstrip().startswith(b"#")]
    name = rel.name.encode("ascii")
    n_leads = len(LEADS)
    if (len(lines) != n_leads + 1 or len(lines[0]) < 4 or lines[0][0] != name
            or lines[0][1] != str(n_leads).encode() or not re.fullmatch(rb"[0-9]{1,7}", lines[0][3])):
        raise ValueError("Expected a single-segment, 12-lead ECG header")
    # WFDB follows filenames and allocation fields in headers; accept only the local paired signal.
    formats = {line[1] for line in lines[1:] if len(line) >= 2 and line[0] == name + b".dat"}
    if len(formats) != 1 or any(len(line) < 2 or line[0] != name + b".dat" for line in lines[1:]):
        raise ValueError("ECG header must reference only its paired signal file")
    fmt = re.fullmatch(rb"16(?:\+([0-9]{1,7}))?", formats.pop())
    if not fmt:
        raise ValueError("Expected a 16-bit ECG signal without skew or extra samples per frame")
    data = _bounded_read(_local_file(root, rel.with_suffix(".dat")), 4 * 1024 * 1024)
    samples = int(lines[0][3])
    if samples < 2 or int(fmt[1] or 0) + samples * n_leads * 2 > len(data):
        raise ValueError("ECG signal file does not contain the declared samples")
    return rel.name, header, data


def _ecg_features(rows, root):
    features = np.full((len(rows), 7 * len(LEADS)), np.nan, dtype=np.float32)
    wanted = defaultdict(list)
    for idx, row in enumerate(rows):
        if row["ecg_path"]:
            wanted[row["ecg_path"]].append(idx)
    if not wanted:
        return features
    root = root.resolve(strict=True)
    with TemporaryDirectory(prefix="repro-ecg-") as scratch:
        scratch = Path(scratch)
        for raw_path, indices in wanted.items():
            name, header, data = _ecg_pair(root, raw_path)
            hea, dat = scratch / f"{name}.hea", scratch / f"{name}.dat"
            hea.write_bytes(header)
            dat.write_bytes(data)
            vector = ecg_features([{"ecg_path": name}], scratch)[0]
            if not np.isfinite(vector).all():
                raise ValueError("Nonfinite or invalid ECG features for a present record")
            features[indices] = vector
            hea.unlink()
            dat.unlink()
    return features


def extract(manifest, data_npy, ecg_root, labevents, out, batch_size=64):
    if out.suffix != ".npz" or batch_size < 1:
        raise ValueError("Extraction needs an .npz output and a positive batch size")
    rows = list(read_rows(manifest, ("hadm_id", "subject_id", "split", "risk", "admittime",
                                     "cxr_StudyDateTime", "ecg_path")))
    if (not rows or len({r["hadm_id"] for r in rows}) != len(rows)
            or any(not r["hadm_id"] or not r["subject_id"] or r["split"] not in SPLITS
                   or r["risk"] not in {"0", "1"} for r in rows)):
        raise ValueError("Expected a nonempty, labeled cohort manifest with unique admissions")
    cxr = _cxr_features(rows, data_npy, batch_size)
    ecg = _ecg_features(rows, ecg_root)
    labs, panels = _lab_features(rows, labevents)
    save_features(out, rows, cxr, ecg, labs, SOURCE, panels)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cohort = commands.add_parser("cohort", help="write the labeled cohort CSV")
    for name in ("symile", "admissions", "transfers", "out"):
        cohort.add_argument(f"--{name}", type=Path, required=True)
    extraction = commands.add_parser(
        "extract", help="write an NPZ in manifest row order",
        description="Extract local features using cached torchvision DenseNet121 IMAGENET1K_V1 weights.",
    )
    for name in ("manifest", "data-npy", "labevents", "out"):
        extraction.add_argument(f"--{name}", type=Path, required=True)
    extraction.add_argument("--ecg-root", type=Path, required=True,
                            help="directory containing unpacked WFDB records at manifest-relative paths")
    extraction.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("--out must be a new file")
    if args.command == "cohort":
        rows, audit = build_cohort(args.symile, args.admissions, args.transfers)
        write_rows(args.out, MANIFEST, rows)
        print(json.dumps(audit, sort_keys=True))
    else:
        extract(args.manifest, args.data_npy, args.ecg_root, args.labevents, args.out, args.batch_size)


if __name__ == "__main__":
    main()
