"""Rates with 95% ranges that span a Wilson interval and a family bootstrap.

  uv run python -m eval.metrics
"""
import random, statistics

Z = 1.96


def wilson(k, n):
    """The 95% Wilson interval for k out of n."""
    if n == 0:
        return (0.0, 1.0)
    p, z2 = k / n, Z * Z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z / (1 + z2 / n) * (p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5
    return (max(0.0, centre - half), min(1.0, centre + half))


def family_bootstrap(counts, reps=10_000, seed=0):
    """The 95% range of sum(k) / sum(n), from `reps` redraws of the families, one (k, n) each."""
    rng = random.Random(seed)
    ratios = []
    for _ in range(reps):
        draw = rng.choices(counts, k=len(counts))
        n = sum(c[1] for c in draw)
        if n:
            ratios.append(sum(c[0] for c in draw) / n)
    if len(ratios) < 2:
        return (0.0, 1.0)
    cuts = statistics.quantiles(ratios, n=40, method="inclusive")
    return (cuts[0], cuts[-1])


def rate(counts):
    """k / n summed over the families' (k, n), with a 95% range that covers both methods' ranges.
    `high_from` names the method that set the high end, the one the gate reads."""
    k, n = sum(c[0] for c in counts), sum(c[1] for c in counts)
    ranges = {"wilson": wilson(k, n), "family bootstrap": family_bootstrap(counts)}
    return {"k": k, "n": n, "rate": k / n if n else None,
            "low": min(r[0] for r in ranges.values()), "high": max(r[1] for r in ranges.values()),
            "high_from": max(ranges, key=lambda m: ranges[m][1])}


if __name__ == "__main__":
    low, high = wilson(5, 10)
    assert (round(low, 4), round(high, 4)) == (0.2366, 0.7634), (low, high)
    assert wilson(0, 381)[1] < 0.01 < wilson(0, 380)[1], "381 is the fewest hits where 0 wrong bounds under 1%"
    clustered = [(5, 5)] * 2 + [(0, 5)] * 98
    r = rate(clustered)
    assert r["high_from"] == "family bootstrap" and r["high"] > wilson(10, 500)[1], r
    assert family_bootstrap(clustered) == family_bootstrap(clustered), "the same seed must give the same range"
    # 3 wrong in 853, 2 of them in one family: the bootstrap's range is wider but tops out lower, so
    # reporting the wider range alone would pass a 1% gate that Wilson fails.
    lopsided = rate([(2, 4), (1, 3)] + [(0, 3)] * 282)
    assert lopsided["high_from"] == "wilson" and lopsided["low"] == 0.0 and lopsided["high"] > 0.01, lopsided
    print("metrics ok:", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
