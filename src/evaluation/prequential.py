"""Contract 3: prequential (test-then-train) evaluation down a time-ordered stream."""
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics

RESULT_COLUMNS = [
    "model_name", "batch_idx", "n", "n_fraud",
    "pr_auc", "roc_auc", "recall_at_p", "fpr", "threshold", "drift_flag",
]


def prequential_eval(model, stream, drift_detector=None, precision_target=0.90,
                     warmup_batches=1, error_threshold=0.5):
    """Warm up on the first warmup_batches batches, then for every later batch: predict -> score -> update.

    drift_detector, if given, needs update(errors) -> bool. For each batch it receives one 0/1
    value per fraudulent transaction (1 = missed, i.e. scored below error_threshold); when it
    fires, the model is refit on the current batch instead of the usual partial_fit.

    Returns one row per evaluated batch; batch_idx counts from the start of the stream, so the
    first evaluated batch is batch_idx == warmup_batches. With a detector, model_name is
    "<model>+<detector>".
    """
    stream = iter(stream)
    warmup = [next(stream) for _ in range(warmup_batches)]
    model.fit(pd.concat([X for X, _ in warmup]), pd.concat([y for _, y in warmup]))

    name = model.name if drift_detector is None else f"{model.name}+{drift_detector.name}"
    rows = []
    for batch_idx, (X, y) in enumerate(stream, start=warmup_batches):
        scores = model.predict_proba(X)
        metrics = compute_metrics(y, scores, precision_target)

        drift = False
        is_fraud = np.asarray(y) == 1
        if drift_detector is not None and is_fraud.any():
            missed = (scores[is_fraud] < error_threshold).astype(int)
            drift = bool(drift_detector.update(missed))

        if drift:
            model.refit(X, y)
        else:
            model.partial_fit(X, y)

        rows.append({
            "model_name": name,
            "batch_idx": batch_idx,
            "n": len(y),
            "n_fraud": int(is_fraud.sum()),
            **metrics,
            "drift_flag": int(drift),
        })

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
