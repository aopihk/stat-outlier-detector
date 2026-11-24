"""Turn "how surprising is this value" into a z-score.

All functions return values on the same scale so different kinds of outliers
can be ranked together: a tail probability p becomes the z with the same
one-sided tail under a standard normal.
"""

import math

import numpy as np
from scipy import stats

P_FLOOR = 1e-300


def _clip(z: float, cap: float) -> float:
    return float(max(-cap, min(cap, z)))


def _tail_z(p: float) -> float:
    return float(stats.norm.isf(max(p, P_FLOOR)))


def count_z(x: float, mu: float, alpha: float, distribution: str, floor: float, cap: float) -> float:
    """Surprise of a count `x` given expected value `mu` and overdispersion `alpha`.

    count:  negative binomial with variance mu + alpha * mu^2 (Poisson if alpha ~ 0),
            converted from tail probability. Handles low-count stats without floors.
    normal: (x - mu) / sd, with sd from the same variance and a hand-set floor.
    """
    mu = max(mu, 1e-3)
    if distribution == "normal":
        sd = max(math.sqrt(mu + alpha * mu * mu), floor)
        return _clip((x - mu) / sd, cap)

    if alpha <= 1e-6:
        dist = stats.poisson(mu)
    else:
        n = 1.0 / alpha
        dist = stats.nbinom(n, n / (n + mu))
    if x > mu:
        return _clip(max(0.0, _tail_z(dist.sf(x - 1))), cap)   # P(X >= x)
    return _clip(min(0.0, -_tail_z(dist.cdf(x))), cap)          # P(X <= x)


def pct_z(makes: int, attempts: int, p: float, concentration: float, distribution: str, cap: float) -> float:
    """Surprise of `makes` out of `attempts` for a shooter with expected rate `p`.

    Beta-binomial allows for normal night-to-night variation beyond pure binomial noise.
    """
    p = min(max(p, 0.01), 0.99)
    n, c = attempts, concentration
    if distribution == "normal":
        sd = math.sqrt(n * p * (1 - p) * (n + c) / (1 + c))
        return _clip((makes - n * p) / sd if sd > 0 else 0.0, cap)

    dist = stats.betabinom(n, p * c, (1 - p) * c)
    if makes > n * p:
        return _clip(max(0.0, _tail_z(dist.sf(makes - 1))), cap)
    return _clip(min(0.0, -_tail_z(dist.cdf(makes))), cap)


def empirical_z(value: float, sorted_values: np.ndarray, tail: str, cap: float) -> tuple[float, float]:
    """Percentile of `value` within a reference distribution, as (z, tail probability).

    tail="high" only scores unusually large values; tail="both" scores either direction.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0, 1.0
    at_or_above = n - np.searchsorted(sorted_values, value, side="left")
    at_or_below = np.searchsorted(sorted_values, value, side="right")
    p_hi = (at_or_above + 1) / (n + 1)
    p_lo = (at_or_below + 1) / (n + 1)
    if tail == "high" or p_hi <= p_lo:
        return _clip(max(0.0, _tail_z(p_hi)), cap), float(p_hi)
    return _clip(min(0.0, -_tail_z(p_lo)), cap), float(p_lo)
