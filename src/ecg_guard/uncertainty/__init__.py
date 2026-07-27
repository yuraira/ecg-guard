"""Single-model uncertainty proxies and selective prediction."""

from ecg_guard.uncertainty.selective import (
    decision_boundary_uncertainty,
    fit_coverage_cutoffs,
    predictive_entropy,
)

__all__ = [
    "decision_boundary_uncertainty",
    "fit_coverage_cutoffs",
    "predictive_entropy",
]
