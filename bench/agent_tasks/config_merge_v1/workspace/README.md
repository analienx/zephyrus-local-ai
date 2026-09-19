# Disposable layered-config merge — coding-agent task

Repair `src/config.py:merge_layers(defaults, file_overrides, env_overrides)`.
All three inputs are dictionaries, with file and environment overrides optional
(`None` or `{}`). Merge recursively: environment overrides file, file overrides
defaults. `None` at any override key means *not specified*, but `0`, `False`
and the empty string are valid explicit values. An unknown key at any nesting
level is an error (`ValueError`), not an ignored or newly created setting.
Nested dictionaries retain defaults for keys that were not overridden.
The return value must not alias or mutate any input at any nesting level;
replace lists as atomic values, copying them. All inputs use string keys.
Do not edit public tests or add dependencies. Only `src/config.py` is writable.
Evaluator-owned tests are outside your accessible workspace.
