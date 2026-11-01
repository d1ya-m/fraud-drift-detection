"""Contract 3: prequential (test-then-train) evaluation down a time-ordered stream."""
from collections import deque

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics, operational_metrics

RESULT_COLUMNS = [
    "model_name", "batch_idx", "n", "n_fraud",
    "pr_auc", "roc_auc", "recall_at_p", "fpr", "threshold",
    "op_recall", "op_precision", "op_fpr", "drift_flag",
]


def prequential_eval(model, stream, drift_detector=None, precision_target=0.90,
                     warmup_batches=1, error_threshold=0.5, label_delay=0, score_sink=None):
    """Warm up on the first warmup_batches batches, then for every later batch: predict -> score -> update.

    Metrics per batch:
      pr_auc, roc_auc               ranking quality
      recall_at_p, fpr, threshold   at the threshold reaching precision_target on THIS batch's labels
                                    (an oracle upper bound: a live system can't pick it this way)
      op_recall, op_precision, op_fpr  at the threshold carried over from the previous batch whose
                                    labels are known (what a live system could actually achieve)

    label_delay: labels of a batch reach the model and the detector only label_delay batches after the
    batch was scored (0 = immediately). Evaluation itself always uses the true labels.

    drift_detector, if given, needs update(errors) -> bool. It receives one 0/1 value per fraudulent
    transaction of a labelled batch (1 = missed: scored below error_threshold when it was scored); when it
    fires, the model is refit on that batch instead of the usual partial_fit. A model that runs its own
    detector can report an alarm through a truthy `last_alarm` attribute after partial_fit.

    score_sink: optional list; each scored batch appends a DataFrame of (batch_idx, row, y, score).

    Returns one row per evaluated batch; batch_idx counts from the start of the stream. With a detector,
    model_name is "<model>+<detector>".
    """
    stream = iter(stream)
    warmup = [next(stream) for _ in range(warmup_batches)]
    model.fit(pd.concat([X for X, _ in warmup]), pd.concat([y for _, y in warmup]))

    name = model.name if drift_detector is None else f"{model.name}+{drift_detector.name}"
    pending = deque()  # scored batches whose labels haven't arrived yet: (X, y, scores)
    live_threshold = np.inf  # latest threshold learnt from labelled data
    rows = []
    for batch_idx, (X, y) in enumerate(stream, start=warmup_batches):
        scores = model.predict_proba(X)
        metrics = compute_metrics(y, scores, precision_target)
        operational = operational_metrics(y, scores, live_threshold)
        if score_sink is not None:
            score_sink.append(pd.DataFrame({"batch_idx": batch_idx, "row": X.index, "y": np.asarray(y), "score": scores}))

        pending.append((X, y, scores, metrics["threshold"]))
        drift = False
        while len(pending) > label_delay:
            X_l, y_l, s_l, thr_l = pending.popleft()  # this batch's labels are now known
            if np.isfinite(thr_l):
                live_threshold = thr_l
            alarm = False
            is_fraud = np.asarray(y_l) == 1
            if drift_detector is not None and is_fraud.any():
                missed = (s_l[is_fraud] < error_threshold).astype(int)
                alarm = bool(drift_detector.update(missed))
            if alarm:
                model.refit(X_l, y_l)
            else:
                model.partial_fit(X_l, y_l)
            drift = drift or alarm or bool(getattr(model, "last_alarm", False))

        rows.append({
            "model_name": name,
            "batch_idx": batch_idx,
            "n": len(y),
            "n_fraud": int(np.sum(y)),
            **metrics,
            **operational,
            "drift_flag": int(drift),
        })

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
