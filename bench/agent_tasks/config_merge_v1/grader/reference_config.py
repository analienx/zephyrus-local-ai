"""Trusted fixture oracle for evaluator calibration; not model-visible."""
from copy import deepcopy


def _merge(base, override):
    if override is None:
        return deepcopy(base)
    if not isinstance(override, dict):
        raise ValueError('override layer must be a dictionary')
    result = deepcopy(base)
    for key, value in override.items():
        if key not in base:
            raise ValueError(f'Unknown configuration key: {key}')
        if value is None:
            continue
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError('Nested override must be a dictionary')
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def merge_layers(defaults, file_overrides, env_overrides):
    if not isinstance(defaults, dict):
        raise ValueError('defaults must be a dictionary')
    return _merge(_merge(defaults, file_overrides), env_overrides)
