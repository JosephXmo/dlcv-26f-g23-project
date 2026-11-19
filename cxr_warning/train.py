import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .cohort import CHEXPERT, SPLITS, read_rows
from .metrics import paired_ap_interval, ranking, select_threshold, summarize


class FusionModel(nn.Module):
    def __init__(self, dimensions, method, width=32):
        super().__init__()
        self.method = method
        self.active = (0,) if method == "cxr" else (1, 2) if method == "clinical" else (0, 1, 2)
        self.encoders = nn.ModuleList(nn.Sequential(nn.Linear(dimensions[idx], width), nn.ReLU())
                                      for idx in self.active)
        fusion_dim = width * len(self.active)
        self.gate = nn.Linear(fusion_dim + len(self.active), len(self.active)) if method == "gate" else None
        risk_dim = width if self.gate is not None else fusion_dim
        self.risk = nn.Sequential(nn.Linear(risk_dim + len(self.active), width), nn.ReLU(), nn.Linear(width, 1))
        self.aux = nn.Linear(width, len(CHEXPERT)) if 0 in self.active else None

    def forward(self, inputs, available):
        mask = available[:, self.active]
        embeddings = torch.stack([encoder(inputs[idx]) for idx, encoder in zip(self.active, self.encoders)], dim=1)
        embeddings = embeddings.masked_fill(~mask.unsqueeze(-1), 0)
        flat = embeddings.flatten(1)
        if self.gate is not None:
            logits = self.gate(torch.cat((flat, mask), dim=1))
            weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(dim=1)
            flat = (embeddings * weights.unsqueeze(-1)).sum(dim=1)
        risk = self.risk(torch.cat((flat, mask), dim=1)).squeeze(1)
        aux = self.aux(embeddings[:, 0]) if self.aux is not None else None
        return risk, aux


def standardize(values, train, append_mask):
    observed = np.isfinite(values)
    train_mask = observed[train]
    train_values = np.where(train_mask, values[train], 0).astype(np.float64)
    counts = np.maximum(train_mask.sum(axis=0), 1)
    mean = train_values.sum(axis=0) / counts
    centered = np.where(train_mask, train_values - mean, 0)
    scale = np.sqrt((centered**2).sum(axis=0) / counts)
    scaled = np.where(observed, (values - mean) / np.where(scale > 1e-6, scale, 1), 0).astype(np.float32)
    return np.concatenate((scaled, observed.astype(np.float32)), axis=1) if append_mask else scaled


def load_data(manifest, features):
    required = ("hadm_id", "subject_id", "split", "risk", "cxr_ViewPosition", *CHEXPERT)
    rows = list(read_rows(manifest, required))
    index = {split: np.array([row["split"] == split for row in rows]) for split in SPLITS}
    admissions = [row["hadm_id"] for row in rows]
    patients = {}
    if len(set(admissions)) != len(rows) or "" in admissions:
        raise ValueError("队列包含重复或空住院标识")
    for row in rows:
        patient, split = row["subject_id"], row["split"]
        if not patient or split not in SPLITS or patients.setdefault(patient, split) != split:
            raise ValueError("队列包含无效 split 或患者跨 split")
    risk = np.array([int(row["risk"]) for row in rows], dtype=np.int64)
    for split in SPLITS:
        if set(risk[index[split]]) != {0, 1}:
            raise ValueError(f"{split} 必须同时含有阳性和阴性。请先审计队列和终点。")
    with np.load(features, allow_pickle=False) as cached:
        feature_ids = cached["hadm_id"].tolist()
        if feature_ids != admissions:
            raise ValueError("特征缓存与队列的 hadm_id 或行顺序不一致")
        raw = [cached[name] for name in ("cxr", "ecg", "labs")]
        source = str(cached["source"].item())
    if any(values.ndim != 2 or len(values) != len(rows) or not values.shape[1] for values in raw):
        raise ValueError("缓存包含无效的特征矩阵")
    available = np.stack([np.isfinite(values).any(axis=1) for values in raw], axis=1)
    if not np.isfinite(raw[0]).all():
        raise ValueError("胸片 embedding 必须存在且有限")
    values = [standardize(array, index["train"], idx == 2) for idx, array in enumerate(raw)]
    findings = np.array([[float(row[name]) if row[name] else np.nan for name in CHEXPERT]
                         for row in rows], dtype=np.float32)
    data = tuple(torch.from_numpy(array) for array in (*values, available, risk.astype(np.float32), findings))
    return rows, index, risk, data, source


@torch.inference_mode()
def predict(model, data, selector, missing=()):
    cxr, ecg, labs, available, _, _ = (array[selector] for array in data)
    if missing:
        available[:, missing] = False
    risk, aux = model((cxr, ecg, labs), available)
    return risk.sigmoid().cpu().numpy(), aux.sigmoid().cpu().numpy() if aux is not None else None


def fit_model(data, split, risk, method, dropout, aux_weight, epochs, seed):
    torch.manual_seed(seed)
    model = FusionModel([array.shape[1] for array in data[:3]], method)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    train_set = TensorDataset(*(array[split["train"]] for array in data))
    loader = DataLoader(train_set, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best_ap, best_epoch, best_state = -1.0, None, None
    for epoch in range(1, epochs + 1):
        model.train()
        for cxr, ecg, labs, available, target, findings in loader:
            if dropout:
                available[:, 1:] &= torch.rand(available[:, 1:].shape) >= dropout
            optimizer.zero_grad()
            logits, aux_logits = model((cxr, ecg, labs), available)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if aux_weight:
                known = (findings == 0) | (findings == 1)
                if known.any():
                    loss = loss + aux_weight * F.binary_cross_entropy_with_logits(aux_logits[known], findings[known])
            loss.backward()
            optimizer.step()
        model.eval()
        scores, _ = predict(model, data, split["val"])
        ap, _ = ranking(risk[split["val"]], scores)
        if ap > best_ap:
            best_ap, best_epoch, best_state = ap, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_epoch, best_ap


def run_experiments(manifest, features, epochs, seed, draws, dropout, aux_weight, specificity):
    torch.set_num_threads(1)
    rows, split, risk, data, source = load_data(manifest, features)
    designs = (
        ("cxr_only", "cxr", 0, 0), ("ecg_labs_only", "clinical", 0, 0),
        ("concat", "concat", dropout, 0), ("gate", "gate", dropout, 0),
        ("gate_no_dropout", "gate", 0, 0), ("gate_multitask", "gate", dropout, aux_weight),
    )
    views = np.array([row["cxr_ViewPosition"] for row in rows])
    patients = np.array([row["subject_id"] for row in rows])
    complete = data[3].numpy().all(axis=1) & split["test"]
    scenarios = (("full_modalities", ()), ("without_ecg", (1,)),
                 ("without_labs", (2,)), ("cxr_only_input", (1, 2)))
    report = {"source": source, "seed": seed, "max_epochs": epochs, "target_specificity": specificity,
              "complete_test_admissions": int(complete.sum()), "models": {}}
    predictions = {}
    for name, method, drop, weight in designs:
        model, epoch, val_ap = fit_model(data, split, risk, method, drop, weight, epochs, seed)
        val, _ = predict(model, data, split["val"])
        threshold = select_threshold(risk[split["val"]], val, specificity)
        test, aux = predict(model, data, split["test"])
        predictions[name] = test
        results = dict(best_epoch=epoch, val_auprc=val_ap, dropout=drop, aux_weight=weight,
                       parameters=sum(parameter.numel() for parameter in model.parameters()),
                       test=summarize(risk[split["test"]], test, threshold))
        results["by_view"] = {view: summarize(risk[split["test"]][mask], test[mask], threshold)
                              for view in ("AP", "PA") if (mask := views[split["test"]] == view).any()}
        if method in {"cxr", "concat", "gate"} and complete.any():
            results["simulated_missingness"] = {
                scenario: summarize(risk[complete], predict(model, data, complete, missing)[0], threshold)
                for scenario, missing in scenarios
            }
        if weight:
            findings = data[-1][split["test"]].numpy()
            results["chexpert"] = {}
            for idx, finding in enumerate(CHEXPERT):
                labels = findings[:, idx]
                known = (labels == 0) | (labels == 1)
                ap, auc = ranking(labels[known], aux[known, idx])
                results["chexpert"][finding] = dict(n=int(known.sum()), positives=int((labels == 1).sum()),
                                                     auprc=ap, auroc=auc)
        report["models"][name] = results
    report["paired_bootstrap"] = {
        f"{variant}_minus_{reference}": paired_ap_interval(
            risk[split["test"]], predictions[variant], predictions[reference], patients[split["test"]], draws, seed,
        )
        for variant, reference in (("concat", "ecg_labs_only"), ("gate", "concat"),
                                   ("gate", "gate_no_dropout"), ("gate_multitask", "gate"))
    }
    return report
