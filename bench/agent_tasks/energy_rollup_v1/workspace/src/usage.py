"""Aggregate renewable-energy device samples into UTC hourly buckets.

Input event: {"device": str, "event_id": str, "when": ISO8601 str, "kwh": number}.
The function returns a list of {"device", "hour_utc", "kwh"}, sorted by
(device, hour_utc). See README for the complete compatibility contract.
"""
from collections import defaultdict
from datetime import datetime


def rollup(events):
    totals = defaultdict(float)
    seen = set()
    for event in events:
        ident = event["event_id"]
        if ident in seen:  # BUG: event IDs are scoped to each device.
            continue
        seen.add(ident)
        when = datetime.fromisoformat(event["when"])
        hour = when.strftime("%Y-%m-%dT%H:00:00Z")  # BUG: not converted to UTC.
        totals[(event["device"], hour)] += float(event["kwh"])
    return [dict(device=dev, hour_utc=hour, kwh=value)
            for (dev, hour), value in sorted(totals.items())]
