"""Replays set B end to end through the live services, then prices and checks the result.

  uv run --env-file .env python -m eval.replay run
  uv run --env-file .env python -m eval.replay report
"""
import datetime, json, random, statistics, sys, time
from collections import Counter
from functools import partial
from pathlib import Path

import boto3

from eval import sweep
from eval.metrics import wilson
from warmstart import clients
from warmstart.cache import Cache, Settings, tidy

SET = "sets/grailqa/set_b.csv"
LOG = Path("results/replay_log.jsonl")
SCOPE = {"model": clients.HAIKU, "prompt": clients.PROMPT, "tools": [], "tier": "basic"}
# US dollars per million tokens, checked 2026-11-03.
PRICES = {
    "haiku": {"input": 0.11, "cache_write": 0.1375, "cache_read": 0.011, "output": 0.55},
    "titan": {"input": 0.026004},
    "jev": {"input": 0.042, "output": 0.0},
}
SOURCES = {
    "checked": "2026-11-03",
    "haiku": "AWS Price List API, AmazonBedrockFoundationModels, eu-west-1, published 2026-10-12: Claude Haiku "
             "5.5, 'Standard' (regional) prices, which the eu. profile pays; global endpoints are 10% less. "
             "Cache writes are the 5-minute ones.",
    "titan": "AWS Price List API, AmazonBedrock, eu-west-1, published 2026-11-02: TitanEmbeddingsV2-Text input "
             "tokens, $0.000026004 per 1,000.",
    "jev": "typesafe.ai: '$42 per billion input tokens'; launch post 'Introducing System One models and Jev': "
           "'Output tokens: FREE'.",
}
PRICE_OF = {"titan": "titan", "personal": "jev", "same": "jev", "haiku": "haiku"}


def frozen():
    bars = json.loads(Path("results/bars.json").read_text())
    return bars, Settings(**{k: bars[k] for k in ("floor", "top", "same_bar", "personal_bar")})


def services(rec, wording):
    """The four live services for one question. Each call notes its tokens and seconds in the question's
    record; the checks also note their score, and "same?" the stored question it was shown."""
    def note(name, fn, *args):
        usage, start = {}, time.perf_counter()
        rec[name] = usage
        value = fn(*args, usage=usage)
        usage["s"] = round(time.perf_counter() - start, 3)
        return value

    def same(stored, question):
        rec["stored"] = stored
        p = note("same", clients.same, stored, question)
        rec["same"]["p"] = p
        return p

    def personal(question):
        p = note("personal", partial(clients.personal, wording=wording), question)
        rec["personal"]["p"] = p
        return p

    return partial(note, "titan", clients.embed), same, personal, partial(note, "haiku", clients.answer)


def run():
    bars, s = frozen()
    rows = sweep.read(SET)
    cache = Cache(s, None, None, None, None)
    cache.invalidate(SCOPE)
    start = time.perf_counter()
    try:
        with LOG.open("w", encoding="utf-8") as f:
            for i, row in enumerate(rows):
                rec = {"i": i, "t": time.time()}
                # Fresh services per question, so each call's tokens land in its own question's record.
                cache.embed, cache.same, cache.personal, cache.answer = services(rec, bars["personal_wording"])
                asked = time.perf_counter()
                text, path = cache.ask(row["question"], SCOPE)
                rec |= {"seconds": round(time.perf_counter() - asked, 3), "path": path,
                        "question": row["question"], "answer": text}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 100 == 0:
                    print(f"{i + 1}: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    finally:
        cache.invalidate(SCOPE)
        cache.pool.shutdown()
    print(f"done: {len(rows)} questions in {(time.perf_counter() - start) / 60:.1f} min")


def cost(service, usage):
    return sum(usage[k] * price for k, price in PRICES[PRICE_OF[service]].items()) / 1e6


def uncached(usage):
    """What a Haiku call would cost with no prompt cache: every prompt token at the input price."""
    flat = {"input": usage["input"] + usage["cache_write"] + usage["cache_read"], "cache_write": 0,
            "cache_read": 0, "output": usage["output"]}
    return cost("haiku", flat)


def spread(xs):
    """p50 and p95 of a list of seconds, with a 95% range on p95 from 10,000 redraws (a bootstrap)."""
    if len(xs) < 2:
        return {"n": len(xs), "seconds": xs}
    p95 = lambda v: statistics.quantiles(v, n=20, method="inclusive")[-1]
    rng = random.Random(0)
    redraws = sorted(p95(rng.choices(xs, k=len(xs))) for _ in range(10_000))
    return {"n": len(xs), "p50": statistics.median(xs), "p95": p95(xs), "p95_range": [redraws[249], redraws[9749]]}


def bedrock_counts(start, end):
    """Bedrock's own counts (CloudWatch) for Haiku and Titan, summed by minute over the window."""
    cw = boto3.client("cloudwatch", region_name=clients.REGION)
    names = {"haiku": (clients.HAIKU, {"input": "InputTokenCount", "cache_write": "CacheWriteInputTokenCount",
                                       "cache_read": "CacheReadInputTokenCount", "output": "OutputTokenCount",
                                       "calls": "Invocations"}),
             "titan": (clients.TITAN, {"input": "InputTokenCount", "calls": "Invocations"})}
    counts = {}
    for service, (model, metrics) in names.items():
        for key, metric in metrics.items():
            points = cw.get_metric_statistics(
                Namespace="AWS/Bedrock", MetricName=metric, Dimensions=[{"Name": "ModelId", "Value": model}],
                StartTime=start, EndTime=end, Period=60, Statistics=["Sum"])["Datapoints"]
            counts[f"{service} {key}"] = int(sum(p["Sum"] for p in points))
    return counts


def report():
    bars, s = frozen()
    rows = sweep.read(SET)
    recs = [json.loads(line) for line in LOG.open(encoding="utf-8")]
    if [r["question"] for r in recs] != [r["question"] for r in rows]:
        raise ValueError(f"{LOG} does not cover {SET} in file order: run the replay again")
    group = {r["question"]: r["group"] for r in rows}
    n = len(recs)

    stored, out = {}, []  # stored: tidied text -> group, for every answer the replay wrote to the cache
    for row, rec in zip(rows, recs):
        path, t = rec["path"], tidy(row["question"])
        served = stored[t] if path == "exact" else group[rec["stored"]] if path == "checked" else None
        p = rec.get("personal", {}).get("p")
        if path == "miss" and p is not None and p <= s.personal_bar:  # the write gate
            stored[t] = row["group"]
        check = (rec["stored"], None, rec["same"]["p"]) if "p" in rec.get("same", {}) else None
        out.append(sweep.Outcome(path, served, check))
    replay = sweep.score(rows, group, out, s) | {"ranges": sweep.ranges(sweep.families(rows, out).values())}
    errors = sum(o.path.endswith("_error") for o in out)

    swept = sweep.run(rows, group, "b gate", s, sweep.personal[bars["personal_wording"], 1])
    swept_ranges = sweep.ranges(sweep.families(rows, swept).values())
    both = list(zip(out, swept))
    memo_personal = sweep.personal[bars["personal_wording"], 1]
    flips = [(rec["personal"]["p"] <= s.personal_bar) != (memo_personal(rec["question"]) <= s.personal_bar)
             for rec in recs if "p" in rec.get("personal", {})]
    agreement = {
        "hit_or_miss_agrees": sum((a.served is None) == (b.served is None) for a, b in both) / n,
        "hit_in_replay_only": sum(a.served is not None and b.served is None for a, b in both),
        "hit_in_sweep_only": sum(a.served is None and b.served is not None for a, b in both),
        "both_hit_same_group": sum(a.served is not None and a.served == b.served for a, b in both),
        "both_hit": sum(a.served is not None and b.served is not None for a, b in both),
        "personal_decision_flips": {"k": sum(flips), "n": len(flips)},
        "sweep_gate": {k: swept_ranges[k] for k in ("hit_rate", "wrong_among_hits", "recall")},
        "ranges_overlap": {k: sweep.overlap(replay["ranges"][k], swept_ranges[k]) for k in swept_ranges},
    }

    calls = {name: [r[name] for r in recs if "s" in r.get(name, {})] for name in PRICE_OF}
    spent = {name: [cost(name, u) for u in us] for name, us in calls.items()}
    mean = {name: statistics.fmean(v) for name, v in spent.items()}
    measured = sum(sum(v) for v in spent.values())
    no_prompt_cache = statistics.fmean(uncached(u) for u in calls["haiku"])
    exact_only = 0  # misses with the exact layer alone: file order, the same "personal?" scores and write gate
    seen = set()
    for rec in recs:
        t = tidy(rec["question"])
        if t not in seen:
            exact_only += 1
            p = rec.get("personal", {}).get("p")
            if p is not None and p <= s.personal_bar:
                seen.add(t)
    per_1000 = {  # each row adds one layer to the one before
        "no_cache": 1000 * no_prompt_cache,
        "prompt_cache": 1000 * mean["haiku"],
        "prompt_cache_and_exact": 1000 * exact_only / n * (mean["haiku"] + mean["personal"]),
        "all_three_measured": 1000 * measured / n,
    }
    per_1000["all_three_without_prompt_cache"] = 1000 / n * (
        measured - sum(spent["haiku"]) + no_prompt_cache * len(calls["haiku"]))

    start = datetime.datetime.fromtimestamp(recs[0]["t"] - 60, datetime.UTC)
    end = datetime.datetime.fromtimestamp(recs[-1]["t"] + recs[-1]["seconds"] + 120, datetime.UTC)
    logged = {f"{service} {k}": sum(u[k] for u in calls[service]) for service in ("haiku", "titan")
              for k in PRICES[service]} | {f"{service} calls": len(calls[service]) for service in ("haiku", "titan")}
    counted = bedrock_counts(start, end)
    check = {k: {"log": logged[k], "bedrock": counted[k], "match": logged[k] == counted[k]} for k in logged}

    paths = Counter(r["path"] for r in recs)
    hits = [o.served is not None for o in out]
    result = {
        "set": "b", "bars": bars, "prices": PRICES | {"sources": SOURCES},
        "window_utc": [start.isoformat(), end.isoformat()],
        "paths": paths, "errors": dict(zip(("k", "n", "low", "high"), (errors, n, *wilson(errors, n)))),
        "replay": replay, "agreement_with_sweep": agreement,
        "cost_per_1000": per_1000,
        "per_call": {name: {"calls": len(us), "mean_cost": mean[name],
                            "p50_s": statistics.median(u["s"] for u in us)} for name, us in calls.items()},
        "latency": {path: spread([r["seconds"] for r in recs if r["path"] == path]) for path in paths}
                   | {"all": spread([r["seconds"] for r in recs]),
                      "haiku_alone": spread([u["s"] for u in calls["haiku"]]),
                      "miss_penalty": spread([r["seconds"] - r["haiku"]["s"] for r in recs if r["path"] == "miss"])},
        "warmup": [{"questions": f"{i + 1}-{min(i + 100, n)}", "hit_rate": statistics.fmean(hits[i:i + 100])}
                   for i in range(0, n, 100)],
        "bedrock_check": check,
    }
    Path("results/replay.json").write_text(json.dumps(result, indent=1) + "\n")

    w, lat = replay["ranges"]["wrong_among_hits"], result["latency"]
    print(f"paths {dict(paths)}; hit rate {replay['all']['hit_rate']:.1%}; wrong among hits {w['k']}/{w['n']} "
          f"(95% up to {w['high']:.2%}); errors {errors}")
    print("cost per 1,000: " + ", ".join(f"{k} ${v:.4f}" for k, v in per_1000.items()))
    print("p50/p95 s: " + ", ".join(f"{k} {v['p50']:.3f}/{v['p95']:.3f}" for k, v in lat.items() if "p50" in v))
    print(f"agreement with sweep: {agreement['hit_or_miss_agrees']:.1%} hit-or-miss, ranges overlap "
          f"{agreement['ranges_overlap']}; personal flips {agreement['personal_decision_flips']}")
    print("bedrock check: " + ", ".join(f"{k} {v['log']}/{v['bedrock']}" for k, v in check.items()))
    paid = sum(m.paid for m in [sweep.embed, *sweep.same.values(), *sweep.personal.values()])
    print(f"memo calls paid during the report: {paid}")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
