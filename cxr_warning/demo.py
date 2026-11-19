from datetime import datetime, timedelta

import numpy as np

from .cohort import CHEXPERT, METADATA, SPLITS, build_cohort, write_rows
from .features import lab_features, save_features


def demo_data(out, seed):
    rng = np.random.default_rng(seed)
    symile = out / "synthetic_symile"
    admit = datetime(2150, 1, 1, 8)
    index = admit + timedelta(hours=36)
    rows, transfers, labevents, embeddings, ecgs = [], [], [], [], []
    for split, size in zip(SPLITS, (320, 100, 160)):
        members = []
        cxr = rng.normal(size=(size, 24)).astype(np.float32)
        ecg = rng.normal(size=(size, 14)).astype(np.float32)
        blood = rng.normal(size=(size, 8)).astype(np.float32)
        planted_risk = cxr[:, 0] + ecg[:, 0] + blood[:, 0] + rng.normal(size=size)
        positives = set(np.argsort(planted_risk)[-size // 5:])
        findings = (cxr[:, :len(CHEXPERT)] > 0.5).astype(int)
        findings[rng.random(findings.shape) < 0.1] = -1
        for idx in range(size):
            admission = str(20000 + len(rows))
            patient = str(10000 + len(rows) // 2)
            ids = dict(hadm_id=admission, subject_id=patient)
            members.append(ids)
            event = index + timedelta(hours=int(rng.integers(1, 49))) if idx in positives else None
            death = event if event and idx % 4 == 0 else None
            discharge = death or index + timedelta(hours=72 if event or idx % 3 else 24)
            rows.append(dict(ids, admittime=admit, dischtime=discharge, deathtime=death,
                             hospital_expire_flag=int(death is not None), cxr_StudyDateTime=index,
                             cxr_ViewPosition="AP" if idx % 2 else "PA", cxr_path=f"{admission}.jpg",
                             ecg_time=admit, ecg_path=f"{admission}", **dict(zip(CHEXPERT, findings[idx]))))
            transfers.append(dict(ids, careunit="Medicine", intime=admit,
                                  outtime=event if event and not death else discharge))
            if event and not death:
                transfers.append(dict(ids, careunit="Medical Intensive Care Unit (MICU)",
                                      intime=event, outtime=discharge))
            for column, value in enumerate(blood[idx]):
                if rng.random() >= 0.15:
                    labevents.append(dict(ids, itemid=column, valuenum=value, valueuom="demo",
                                          charttime=admit, storetime=admit + timedelta(hours=1)))
            embeddings.append(cxr[idx])
            ecgs.append(ecg[idx])
        if split == "test":
            candidates = [dict(candidate, label_hadm_id=query["hadm_id"], label=match)
                          for idx, query in enumerate(members)
                          for candidate, match in ((query, 1), (members[(idx + 1) % size], 0))]
            write_rows(symile / "test.csv", ("hadm_id", "subject_id", "label_hadm_id", "label"), candidates)
        else:
            write_rows(symile / f"{split}.csv", ("hadm_id", "subject_id"), members)
    write_rows(symile / "symile_mimic_data.csv", (*METADATA, *CHEXPERT), rows)
    write_rows(out / "synthetic_transfers.csv", transfers[0].keys(), transfers)
    write_rows(out / "synthetic_labevents.csv", labevents[0].keys(), labevents)
    manifest, audit = build_cohort(symile, out / "synthetic_transfers.csv")
    labs, panels = lab_features(manifest, out / "synthetic_labevents.csv")
    input_order = {row["hadm_id"]: idx for idx, row in enumerate(rows)}
    indices = [input_order[row["hadm_id"]] for row in manifest]
    cache = out / "features.npz"
    save_features(cache, manifest, np.array(embeddings)[indices], np.array(ecgs)[indices], labs, "synthetic", panels)
    return manifest, audit, cache
