from collections import Counter, defaultdict
from datetime import timedelta

import numpy as np

from .cohort import read_rows, timestamp

LAB_COLUMNS = ("subject_id", "hadm_id", "itemid", "valueuom", "valuenum", "charttime", "storetime")
LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
WEIGHTS = "densenet121-res224-chex"


def lab_features(rows, path, limit=50):
    by_patient = defaultdict(list)
    for idx, row in enumerate(rows):
        admitted = timestamp(row["admittime"])
        index = timestamp(row["cxr_StudyDateTime"])
        by_patient[row["subject_id"]].append((idx, row["hadm_id"], admitted, index))

    selected = [dict() for _ in rows]
    for lab in read_rows(path, LAB_COLUMNS):
        candidates = by_patient.get(lab["subject_id"])
        if not candidates or not lab["valuenum"] or not lab["itemid"]:
            continue
        chart, store = timestamp(lab["charttime"]), timestamp(lab["storetime"])
        if chart is None or store is None:
            continue
        value = float(lab["valuenum"])
        if not np.isfinite(value):
            continue
        key = (lab["itemid"], lab["valueuom"].strip())
        for idx, admission, admitted, index in candidates:
            if lab["hadm_id"] != admission:
                if lab["hadm_id"] or not chart < admitted:
                    continue
            if not (admitted - timedelta(hours=24) <= chart <= admitted + timedelta(hours=24)
                    and chart <= store <= index):
                continue
            previous = selected[idx].get(key)
            if previous is None or (chart, store) < previous[:2]:
                selected[idx][key] = (chart, store, value)

    frequencies = Counter(key for row, values in zip(rows, selected) if row["split"] == "train" for key in values)
    panels = sorted(frequencies, key=lambda key: (-frequencies[key], key))[:limit]
    if not panels:
        raise ValueError("训练队列没有符合时间条件的数值血检")
    labs = np.full((len(rows), len(panels)), np.nan, dtype=np.float32)
    for idx, values in enumerate(selected):
        for column, panel in enumerate(panels):
            if panel in values:
                labs[idx, column] = values[panel][2]
    return labs, panels


def ecg_features(rows, root):
    import wfdb

    features = np.full((len(rows), 7 * len(LEADS)), np.nan, dtype=np.float32)
    for idx, row in enumerate(rows):
        if not row["ecg_path"]:
            continue
        record_path = root / row["ecg_path"]
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal
        names = {name.upper(): column for column, name in enumerate(record.sig_name)}
        if signal is None or signal.shape[0] < 2 or any(lead not in names for lead in LEADS):
            continue
        signal = signal[:, [names[lead] for lead in LEADS]]
        if not np.isfinite(signal).all() or not np.any(signal):
            continue
        quantiles = np.quantile(signal, (0.05, 0.5, 0.95), axis=0)
        features[idx] = np.concatenate((
            signal.mean(axis=0), signal.std(axis=0), *quantiles,
            np.sqrt(np.mean(signal**2, axis=0)), np.abs(np.diff(signal, axis=0)).mean(axis=0),
        ))
    return features


def cxr_features(rows, root, batch_size, device, cache_dir):
    import torch
    import torchxrayvision as xrv
    from PIL import Image

    backbone = xrv.models.DenseNet(weights=WEIGHTS, cache_dir=str(cache_dir)).requires_grad_(False).eval().to(device)
    crop, resize = xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            images = []
            for row in rows[start:start + batch_size]:
                with Image.open(root / row["cxr_path"]) as image:
                    pixels = np.asarray(image.convert("L"))
                pixels = xrv.utils.normalize(pixels, 255, reshape=True)
                images.append(resize(crop(pixels)))
            inputs = torch.from_numpy(np.stack(images).astype(np.float32)).to(device)
            batches.append(backbone.features2(inputs).cpu().numpy())
    return np.concatenate(batches)


def save_features(path, rows, cxr, ecg, labs, source, lab_panels=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, hadm_id=np.array([row["hadm_id"] for row in rows]),
                        cxr=cxr, ecg=ecg, labs=labs, source=np.array(source), lab_panels=np.array(lab_panels))
