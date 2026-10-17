"""Contract 2: every model exposes the same four methods so the harness can treat them alike."""


class BaseDetector:
    name = "base"

    def fit(self, X, y):
        """Batch training / warm-up."""
        raise NotImplementedError

    def partial_fit(self, X, y):
        """Per-batch update. Default is a no-op, i.e. a frozen model."""
        return self

    def predict_proba(self, X):
        """P(fraud) in [0, 1], shape (n,)."""
        raise NotImplementedError

    def refit(self, X, y):
        """Adapt after a detected drift. Default: full re-fit on the given data."""
        return self.fit(X, y)
