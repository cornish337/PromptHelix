from __future__ import annotations

from collections.abc import Mapping
from typing import Any

def update_settings(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """
    Recursively deep-merges `overrides` into a copy of `base`.

    - Nested mappings are merged by recursing.
    - Scalar or non-mapping values in `overrides` overwrite those in `base`.
    - If `overrides` is None or empty, returns a shallow copy of `base`.

    Args:
        base:      Original settings mapping.
        overrides: Overrides to apply.

    Returns:
        A new dict with merged settings.
    """
    # Start with a shallow copy so we don’t mutate the input
    merged: dict[str, Any] = dict(base)

    if not overrides:
        return merged

    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            # Recurse for nested mappings
            merged[key] = update_settings(merged[key], value)
        else:
            # Override or add
            merged[key] = value

    return merged
