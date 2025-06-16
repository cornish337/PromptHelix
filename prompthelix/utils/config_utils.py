from __future__ import annotations

from collections.abc import Mapping
from typing import Any, MutableMapping


def update_settings(base: MutableMapping[str, Any], overrides: Mapping[str, Any] | None) -> MutableMapping[str, Any]:
    """Recursively merge override values into ``base``.

    Parameters
    ----------
    base:
        Original settings dictionary to update.
    overrides:
        Dictionary with values that should override those in ``base``.

    Returns
    -------
    MutableMapping[str, Any]
        The updated ``base`` dictionary.
    """
    if not overrides:
        return base

    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], Mapping)
            and isinstance(value, Mapping)
        ):
            update_settings(base[key], value)
        else:
            base[key] = value
    return base
