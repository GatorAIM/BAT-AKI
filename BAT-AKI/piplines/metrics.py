import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
)

def compute_bootstrap_ci(labels, probs, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    metrics = {k: [] for k in ["accuracy", "precision", "recall", "auroc", "auprc"]}

    for _ in range(n_bootstraps):
        idx = rng.choice(len(labels), len(labels), replace=True)
        y = labels[idx]
        p = probs[idx]

        if len(np.unique(y)) < 2:
            continue

        pred = (p >= 0.5).astype(int)

        metrics["accuracy"].append(accuracy_score(y, pred))
        metrics["precision"].append(precision_score(y, pred, zero_division=0))
        metrics["recall"].append(recall_score(y, pred, zero_division=0))
        metrics["auroc"].append(roc_auc_score(y, p))
        metrics["auprc"].append(average_precision_score(y, p))

    return {
        k: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5))
        for k, v in metrics.items()
    }
