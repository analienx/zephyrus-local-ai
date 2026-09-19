"""Merge settings from defaults, file and environment (see README contract)."""


def merge_layers(defaults, file_overrides, env_overrides):
    merged = dict(defaults)
    if file_overrides:
        merged.update(file_overrides)  # BUG: shallow and accepts unknown keys.
    if env_overrides:
        # BUG: falsy values are valid overrides, not missing settings.
        merged.update({k: v for k, v in env_overrides.items() if v})
    return merged
