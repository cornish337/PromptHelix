"""Minimal helper for logging metrics to Weights & Biases."""

import os

try:
    import wandb  # type: ignore
except Exception:  # Package might not be installed
    wandb = None

_run = None


def _get_run():
    """Initialize a W&B run if possible."""
    global _run
    if _run is None and wandb is not None and os.getenv("WANDB_API_KEY"):
        _run = wandb.init(project="prompthelix", reinit=True)
    return _run


def log_metrics(metrics: dict) -> None:
    """Log metrics to W&B if configured."""
    run = _get_run()
    if run is not None:
        run.log(metrics)
