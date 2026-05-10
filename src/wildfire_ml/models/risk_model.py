"""Wildfire risk scoring model (XGBoost/CatBoost).

Implementation: Phase 1 Sprint 4.
"""


class RiskModel:
    """XGBoost-based wildfire risk score model.

    Phase 0.5: skeleton.
    """

    def __init__(self) -> None:
        self.model = None

    def fit(self, X, y):
        raise NotImplementedError("Implement in Phase 1 Sprint 4")

    def predict(self, X):
        raise NotImplementedError("Implement in Phase 1 Sprint 4")
