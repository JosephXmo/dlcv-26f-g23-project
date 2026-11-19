import numpy as np


def operating_curve(labels, scores):
    order = np.argsort(-scores)
    ranked_labels, ranked_scores = labels[order], scores[order]
    ends = np.flatnonzero(np.r_[ranked_scores[1:] != ranked_scores[:-1], True])
    tp = np.cumsum(ranked_labels)[ends]
    fp = ends + 1 - tp
    return tp, fp, ranked_scores[ends]


def ranking(labels, scores):
    if len(np.unique(labels)) < 2:
        return None, None
    tp, fp, _ = operating_curve(labels, scores)
    tpr, fpr = np.r_[0, tp / tp[-1]], np.r_[0, fp / fp[-1]]
    ap = np.sum(np.diff(tpr) * (tp / (tp + fp)))
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    return float(ap), float(auc)


def select_threshold(labels, scores, specificity):
    tp, fp, thresholds = operating_curve(labels, scores)
    allowed = np.flatnonzero((1 - fp / fp[-1] >= specificity) & (tp > 0))
    return float(thresholds[allowed[-1]]) if len(allowed) else None


def summarize(labels, scores, threshold):
    ap, auc = ranking(labels, scores)
    positives = labels == 1
    called = scores >= threshold if threshold is not None else np.zeros_like(positives)
    bins = []
    for idx in range(5):
        in_bin = (np.minimum((scores * 5).astype(int), 4) == idx)
        if in_bin.any():
            bins.append(dict(bin=idx, n=int(in_bin.sum()), mean_score=float(scores[in_bin].mean()),
                             observed_rate=float(labels[in_bin].mean())))
    return dict(n=len(labels), positives=int(positives.sum()), auprc=ap, auroc=auc,
                brier=float(np.mean((labels - scores)**2)), threshold_from_validation=threshold,
                sensitivity=float(called[positives].mean()) if positives.any() else None,
                specificity=float((~called[~positives]).mean()) if (~positives).any() else None,
                calibration=bins)


def paired_ap_interval(labels, gate, concat, patients, draws, seed):
    point_gate, _ = ranking(labels, gate)
    point_concat, _ = ranking(labels, concat)
    groups = [np.flatnonzero(patients == patient) for patient in np.unique(patients)]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        idx = np.concatenate([groups[group] for group in rng.integers(0, len(groups), len(groups))])
        gate_ap, _ = ranking(labels[idx], gate[idx])
        if gate_ap is not None:
            concat_ap, _ = ranking(labels[idx], concat[idx])
            differences.append(gate_ap - concat_ap)
    return dict(delta_auprc=point_gate - point_concat if point_gate is not None else None,
                ci_95=np.quantile(differences, (0.025, 0.975)).tolist() if differences else None,
                valid_draws=len(differences), requested_draws=draws)
