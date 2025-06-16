import collections.abc

def update_settings(base_settings: dict, override_settings: dict) -> dict:
    """
    Recursively merges override_settings into base_settings.

    For nested dictionaries, it performs a deep merge. For other types,
    the value from override_settings takes precedence if the key exists
    in both. New keys from override_settings are added to base_settings.

    Args:
        base_settings: The base dictionary to update.
        override_settings: The dictionary with overrides.

    Returns:
        A new dictionary containing the merged settings.
    """
    # Start with a copy of base_settings to avoid modifying the original
    merged = base_settings.copy()

    if override_settings is None: # Handle case where override_settings might be None
        return merged

    for key, value in override_settings.items():
        if isinstance(value, collections.abc.Mapping) and isinstance(merged.get(key), collections.abc.Mapping):
            merged[key] = update_settings(merged[key], value)
        else:
            merged[key] = value
    return merged
