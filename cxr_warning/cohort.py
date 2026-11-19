import csv
import gzip
from collections import Counter, defaultdict
from datetime import datetime, timedelta

SPLITS = ("train", "val", "test")
HORIZON = timedelta(hours=48)
CHEXPERT = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
)
ICUS = {
    "Cardiac Vascular Intensive Care Unit (CVICU)", "Coronary Care Unit (CCU)",
    "Medical Intensive Care Unit (MICU)", "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)", "Surgical Intensive Care Unit (SICU)",
    "Trauma SICU (TSICU)", "Neonatal Intensive Care Unit (NICU)",
}
WARDS = {
    "Cardiac Surgery", "Cardiology", "Hematology/Oncology", "Medicine",
    "Medicine/Cardiology", "Med/Surg", "Med/Surg/GYN", "Med/Surg/Trauma",
    "Medical/Surgical (Gynecology)", "Neurology", "Surgery", "Surgery/Trauma",
    "Surgery/Pancreatic/Biliary/Bariatric", "Thoracic Surgery", "Transplant", "Vascular",
}
METADATA = (
    "hadm_id", "subject_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag",
    "cxr_StudyDateTime", "cxr_ViewPosition", "cxr_path", "ecg_time", "ecg_path",
)
MANIFEST = (
    "hadm_id", "subject_id", "split", "admittime", "cxr_StudyDateTime", "cxr_ViewPosition",
    "cxr_path", "ecg_path", "index_ward", "risk", "icu_48h", "death_48h", "lead_hours",
    *CHEXPERT,
)


def read_rows(path, required):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} 缺少列 {', '.join(sorted(missing))}")
        yield from reader


def write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def timestamp(value):
    return datetime.fromisoformat(value) if value else None


def split_membership(symile):
    membership = {}
    seen_patients = {}
    for split in SPLITS:
        required = ("hadm_id", "subject_id", "label_hadm_id", "label") if split == "test" else (
            "hadm_id", "subject_id",
        )
        queries = set()
        positives = set()
        for row in read_rows(symile / f"{split}.csv", required):
            admission = row["hadm_id"]
            patient = row["subject_id"]
            if split == "test":
                queries.add(row["label_hadm_id"])
                if row["label"] == "0":
                    continue
                if row["label"] != "1" or row["label_hadm_id"] != admission:
                    raise ValueError("test.csv 中存在无效的检索阳性行")
                positives.add(admission)
            if not admission or not patient or admission in membership:
                raise ValueError(f"{split} 存在空标识或重复住院")
            if seen_patients.setdefault(patient, split) != split:
                raise ValueError("同一患者出现在多个 split")
            membership[admission] = (split, patient)
        if split == "test" and queries != positives:
            raise ValueError("test.csv 的检索 query 未各自对应一条阳性行")
    return membership


def build_cohort(symile, transfers, extra_wards=()):
    membership = split_membership(symile)
    episodes = defaultdict(list)
    for row in read_rows(transfers, ("hadm_id", "subject_id", "careunit", "intime", "outtime")):
        admission = row["hadm_id"]
        if admission in membership and row["careunit"]:
            if row["subject_id"] != membership[admission][1]:
                raise ValueError("MIMIC transfers 与 Symile 的患者标识不一致")
            episodes[admission].append((row["careunit"], timestamp(row["intime"]), timestamp(row["outtime"])))

    wards = WARDS.union(extra_wards) - ICUS
    counts = {split: Counter() for split in SPLITS}
    locations = {split: Counter() for split in SPLITS}
    seen = set()
    manifest = []
    lead_times = defaultdict(list)
    for row in read_rows(symile / "symile_mimic_data.csv", (*METADATA, *CHEXPERT)):
        admission = row["hadm_id"]
        if admission not in membership:
            continue
        split, patient = membership[admission]
        if admission in seen or row["subject_id"] != patient:
            raise ValueError("Symile 主表存在重复住院或患者标识不一致")
        seen.add(admission)
        stats = counts[split]
        stats["input_admissions"] += 1
        admit, discharge, index, death, ecg = map(timestamp, (
            row["admittime"], row["dischtime"], row["cxr_StudyDateTime"], row["deathtime"], row["ecg_time"],
        ))
        if not (admit and discharge and index and admit <= index < discharge):
            stats["excluded_cxr_outside_stay"] += 1
            continue
        if not row["cxr_path"] or row["cxr_ViewPosition"] not in {"AP", "PA"}:
            stats["excluded_missing_frontal_cxr"] += 1
            continue
        expired = row["hospital_expire_flag"]
        if (expired not in {"0", "1"} or (expired == "1") != (death is not None)
                or (death is not None and not index < death <= discharge)):
            stats["excluded_inconsistent_death_or_dead_at_index"] += 1
            continue

        stays = episodes[admission]
        active = {unit for unit, start, end in stays if start and end and start <= index < end}
        locations[split].update(active)
        icu_starts = [start for unit, start, _ in stays if unit in ICUS and start]
        if any(start <= index for start in icu_starts):
            stats["excluded_icu_at_or_before_index"] += 1
            continue
        if len(active) != 1 or not active.issubset(wards):
            stats["excluded_no_unambiguous_ward_at_index"] += 1
            continue
        if any(unit in ICUS and start is None for unit, start, _ in stays):
            stats["excluded_icu_missing_intime"] += 1
            continue

        event_end = min(index + HORIZON, discharge)
        icu_event = min((start for start in icu_starts if start <= event_end), default=None)
        death_event = death if death and death <= event_end else None
        first_event = min((event for event in (icu_event, death_event) if event), default=None)
        lead = (first_event - index).total_seconds() / 3600 if first_event else None
        if lead is not None:
            lead_times[split].append(lead)
        stats["eligible"] += 1
        stats["risk_positive"] += first_event is not None
        stats["icu_positive"] += icu_event is not None
        stats["death_positive"] += death_event is not None
        stats["early_alive_discharge_without_event"] += (
            discharge < index + HORIZON and death is None and first_event is None
        )
        if not (ecg and ecg <= index and row["ecg_path"]):
            row["ecg_path"] = ""
            stats["ecg_unavailable_at_index"] += 1
        manifest.append(dict(row, split=split, index_ward=next(iter(active)),
                             risk=int(first_event is not None), icu_48h=int(icu_event is not None),
                             death_48h=int(death_event is not None), lead_hours=lead))

    if seen != set(membership):
        raise ValueError(f"Symile 主表缺少 {len(membership) - len(seen)} 条 split 住院记录")
    audit = {
        split: dict(counts[split], eligible_patients=len({
            row["subject_id"] for row in manifest if row["split"] == split
        }), lead_hours_histogram={
            f"({lower},{upper}]": sum(lower < lead <= upper for lead in lead_times[split])
            for lower, upper in ((0, 2), (2, 6), (6, 12), (12, 24), (24, 48))
        }, index_locations=dict(locations[split]))
        for split in SPLITS
    }
    return manifest, audit
