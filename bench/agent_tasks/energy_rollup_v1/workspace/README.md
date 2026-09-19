# Disposable device-energy rollup — coding-agent task

Repair `src/usage.py:rollup(events)` without changing the public signature.
Each event is a dictionary with `device`, `event_id`, `when`, and `kwh`.
Device/event IDs must be nonempty strings; `when` is ISO-8601 with an explicit
UTC offset or `Z`; `kwh` is a finite nonnegative int/float (bool is invalid).
Raise `ValueError` for malformed input, including missing keys or bad dates.
Duplicate IDs are ignored **within one device**; two devices can share an ID.
Group the first valid sample per `(device, event_id)` by the corresponding
**UTC hour**, not the event's source timezone. Return deterministically sorted
`{"device": str, "hour_utc": "YYYY-MM-DDTHH:00:00Z", "kwh": float}` rows.
Do not modify public tests, introduce dependencies or access the network.

The workspace is disposable. Only `src/usage.py` is writable by the agent.
The evaluator's private acceptance cases are outside this workspace and must
never be supplied to the model or accepted as tool-modified artifacts.
