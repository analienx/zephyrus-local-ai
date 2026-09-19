"""Independent mathematical reference for one-position speculative verification.

Not a binding to llama.cpp, a GPU test, or a production sampler. The output-law
checks are algebraic and use only the actual post-sampling p and proposal q.
"""
from __future__ import annotations

import math
from typing import Sequence


def validate(p: Sequence[float], q: Sequence[float]) -> None:
    if not p or len(p) != len(q):
        raise ValueError("p and q must be nonempty and cover identical token IDs")
    for values in (p, q):
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("negative or non-finite probability")
        if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("unnormalized distribution")


def coupled_acceptance(p_d: float, q_d: float) -> tuple[float, float]:
    """Return total a and *conditional extra* acceptance after x~p != d.

    q_d=0 cannot describe a proposal genuinely drawn from q.
    """
    if not (0 <= p_d <= 1 and 0 < q_d <= 1):
        raise ValueError("invalid proposed token probability")
    a = min(1.0, p_d / q_d)
    if p_d == 1.0:
        return a, 0.0
    extra = (a - p_d) / (1.0 - p_d)
    if not (-1e-12 <= extra <= 1.0 + 1e-12):
        raise AssertionError("invalid conditional acceptance")
    return a, min(1.0, max(0.0, extra))


def one_position_output(p: Sequence[float], q: Sequence[float]) -> tuple[list[float], float]:
    """Compute the exact output law by enumerating all draft proposals.

    Proposed token d~q is accepted with min(1,p[d]/q[d]). On rejection sample
    residual max(p-q,0); no draft token outside q's support can be accepted.
    """
    validate(p, q)
    accepted = [min(pi, qi) for pi, qi in zip(p, q)]
    accept_mass = sum(accepted)
    reject_mass = 1.0 - accept_mass
    if reject_mass < -1e-12:
        raise AssertionError("over-accepted proposals")
    if reject_mass <= 1e-12:
        return accepted, accept_mass
    residual = [max(0.0, pi - qi) for pi, qi in zip(p, q)]
    norm = sum(residual)
    if not math.isclose(norm, reject_mass, rel_tol=0, abs_tol=1e-10):
        raise AssertionError("residual must equal rejection probability")
    return [ai + reject_mass * ri / norm for ai, ri in zip(accepted, residual)], accept_mass
