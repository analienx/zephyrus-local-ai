"""Trusted fixture oracle for evaluator self-tests. NEVER expose to candidate agent."""
from collections import defaultdict
from datetime import datetime, timezone
from math import isfinite


def rollup(events):
    if not isinstance(events, (list, tuple)):
        raise ValueError('events must be a sequence')
    sums = defaultdict(float)
    seen = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('event must be a mapping')
        try:
            device, ident, when_text, value = (event[k] for k in
                                               ('device', 'event_id', 'when', 'kwh'))
            if not isinstance(device, str) or not device or not isinstance(ident, str) or not ident:
                raise ValueError('missing device/id')
            if not isinstance(when_text, str) or not when_text:
                raise ValueError('missing timestamp')
            when = datetime.fromisoformat(when_text.replace('Z', '+00:00'))
            if when.tzinfo is None or when.utcoffset() is None:
                raise ValueError('timestamp requires explicit timezone')
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError('kwh must be finite and nonnegative')
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError('malformed event') from exc
        key = (device, ident)
        if key in seen:
            continue
        seen.add(key)
        hour = when.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:00:00Z')
        sums[(device, hour)] += float(value)
    return [dict(device=d, hour_utc=h, kwh=n)
            for (d, h), n in sorted(sums.items())]
