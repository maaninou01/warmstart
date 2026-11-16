"""Runs a question set through the real engine and Redis once per bar, and scores every hit.

  uv run --env-file .env python -m eval.sweep a
  uv run --env-file .env python -m eval.sweep b
  uv run --env-file .env python -m eval.sweep gate
"""
import csv, json, random, statistics, sys, time
from collections import Counter, defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import redis

from eval.metrics import rate
from warmstart import clients
from warmstart.cache import Cache, Settings, connect, create_index, tidy

BARS = [round(0.60 + i / 100, 2) for i in range(41)]  # 1% of questions sit only 0.58-0.68 from their own group
NO_TOP = 1.01  # a top bar above 1 means no top zone
BASE = Settings(floor=NO_TOP, top=NO_TOP)
TOP_HITS = 381  # the fewest hits for which 0 wrong bounds the error rate under 1% (Wilson)
WORKERS = 8
STRICT = ("v2", 0.20)  # the strict "personal?" baseline: wording and bar
REVIEW = Path("results/jev_review.csv")
REVIEW_SAMPLE = 100


def memo(name, fn, version, call):
    """Call 1 is the score the engine uses; call 2 asks again, to measure the wobble."""
    again = "" if call == 1 else "_again"
    return clients.Memo(f"memo/{name}{again}.jsonl", fn, version + again)


embed = clients.Memo("memo/titan.jsonl", clients.embed, clients.VERSIONS["embed"])
same = {c: memo("jev_same", clients.same, clients.VERSIONS["same"], c) for c in (1, 2)}
personal = {(w, c): memo(f"jev_personal_{w}", partial(clients.personal, wording=w),
                         clients.VERSIONS[f"personal {w}"], c) for w in clients.PERSONAL for c in (1, 2)}
Outcome = namedtuple("Outcome", "path served check")  # check: (stored question, similarity, "same?" score)


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fetch(fn, *columns):
    """fn over every row of the columns, 8 calls at a time; any failure raises."""
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(fn, *columns))


def wobble(first, second, *columns):
    """How far a second identical call moves a score: the 99th percentile of the gaps."""
    gaps = [abs(a - b) for a, b in zip(fetch(first, *columns), fetch(second, *columns))]
    return statistics.quantiles(gaps, n=100)[-1]


def run(rows, group, name, s, personal_fn):
    """One pass in file order, from an empty namespace of its own. Any failure path stops it."""
    checks = {}

    def checked_same(stored, question):
        p = same[1](stored, question)
        checks[question] = (stored, sum(a * b for a, b in zip(embed(stored), embed(question))), p)
        return p

    cache = Cache(s, embed, checked_same, personal_fn, lambda question, scope: group[question])
    scope = {"model": clients.HAIKU, "prompt": f"sweep {name}", "tools": [], "tier": "basic"}
    cache.invalidate(scope)
    try:
        out = []
        for r in rows:
            served, path = cache.ask(r["question"], scope)
            if path.endswith("_error"):
                raise RuntimeError(f"pass {name} stopped: {path} on {r['question']!r}")
            out.append(Outcome(path, None if path == "miss" else served, checks.get(r["question"])))
        return out
    finally:
        cache.invalidate(scope)
        cache.pool.shutdown()


def score(rows, group, out, s):
    """A pass's numbers against the labels, overall and per kind, and how Jev's checks went."""
    def numbers(pairs):
        hits = [(r, o) for r, o in pairs if o.served is not None]
        wrong = sum(o.served != r["group"] for r, o in hits)
        expected = sum(r["expected"] == "hit" for r, _ in pairs)
        return {"questions": len(pairs), "hits": len(hits), "wrong": wrong,
                "hit_rate": len(hits) / len(pairs), "wrong_among_hits": wrong / len(hits) if hits else 0.0,
                "wrong_among_all": wrong / len(pairs), "recall": (len(hits) - wrong) / expected if expected else None}

    pairs = list(zip(rows, out))
    checks = [(o.check[2] >= s.same_bar, group[o.check[0]] == r["group"]) for r, o in pairs if o.check]
    return {
        "paths": Counter(o.path for o in out),
        "wrong_by_path": Counter(o.path for r, o in pairs if o.served not in (None, r["group"])),
        "all": numbers(pairs),
        "kinds": {k: numbers([p for p in pairs if p[0]["kind"] == k]) for k in sorted({r["kind"] for r in rows})},
        "jev": {"checks": len(checks),
                "wrong_passes": sum(passed and not good for passed, good in checks),
                "good_blocked": sum(good and not passed for passed, good in checks)},
    }


def curves(rows, group, name, jev, modes, pairs=None):
    """One pass per mode and bar, 8 at a time. `pairs` collects every checked pair and whether it is good."""
    def one(job):
        mode, bar = job
        s = replace(BASE, floor=bar, top=bar if mode == "alone" else NO_TOP,
                    same_bar=jev["same_bar"], personal_bar=jev["personal_bar"])
        out = run(rows, group, f"{name} {mode} {bar}", s, personal[jev["personal_wording"], 1])
        if pairs is not None:
            pairs.update({(o.check[0], r["question"]): group[o.check[0]] == r["group"]
                          for r, o in zip(rows, out) if o.check})
        p = {"mode": mode, "bar": bar} | score(rows, group, out, s)
        print(f"{mode} {bar}: hits {p['all']['hits']}, wrong {p['all']['wrong']}", flush=True)
        return p

    return fetch(one, [(m, b) for m in modes for b in BARS])


def families(rows, out):
    """A pass's counts per family: questions, expected hits, hits and wrong hits."""
    fam = defaultdict(Counter)
    for r, o in zip(rows, out):
        f = fam[r["family"]]
        f["questions"] += 1
        f["expected"] += r["expected"] == "hit"
        f["hits"] += o.served is not None
        f["wrong"] += o.served not in (None, r["group"])
    return fam


def ranges(fam):
    """Rates over the families' counts, each with the wider of its two 95% ranges, families redrawn whole."""
    rates = {"hit_rate": ("hits", "questions"), "wrong_among_hits": ("wrong", "hits"),
             "wrong_among_all": ("wrong", "questions")}
    result = {name: rate([(f[k], f[n]) for f in fam]) for name, (k, n) in rates.items()}
    result["recall"] = rate([(f["hits"] - f["wrong"], f["expected"]) for f in fam])
    return result


def personal_set(split, stages=("before scoring", "after scoring")):
    """The Haiku-written messages of one split, with the labels settled in sets/personal/review.csv
    at the given stages: before any Jev score was seen, or after (only messages Jev got wrong)."""
    settled = {r["question"]: r["label"] for r in read("sets/personal/review.csv") if r["stage"] in stages}
    return [r | {"personal": settled.get(r["question"], r["personal"])}
            for r in read("sets/personal/messages.csv") if r["split"] == split]


def judge(wording, bar, msgs):
    """What a "personal?" wording and bar do to labelled messages: personal ones shared (leaks), general
    ones kept, unsure ones shared. Ranges redraw whole batches: the messages Haiku wrote in one call."""
    p = personal[wording, 1]

    def share(label):
        batches = defaultdict(lambda: [0, 0])
        for r in msgs:
            if r["personal"] == label:
                b = batches[r["business"], r["kind"]]
                b[0] += p(r["question"]) <= bar
                b[1] += 1
        return rate(list(batches.values()))

    fetch(p, [r["question"] for r in msgs])
    return {"bar": bar, "leaks": share("yes"), "general_kept": share("no"), "unsure_shared": share("unsure")}


def tune_personal(grail):
    """Per wording, the bar by the rule and what it keeps on the tuning businesses and set A (all
    general), next to the strict baseline."""
    msgs = [r for r in personal_set("tune") if r["personal"] != "unsure"]

    def assess(wording, bar):
        return {"wording": wording, "tune": judge(wording, bar, msgs),
                "grailqa_kept": {"k": sum(personal[wording, 1](t) <= bar for t in grail), "n": len(grail)}}

    report = {}
    for w in clients.PERSONAL:
        wob = wobble(personal[w, 1], personal[w, 2], [r["question"] for r in msgs] + grail)
        lowest = min(personal[w, 1](r["question"]) for r in msgs if r["personal"] == "yes")
        report[w] = {"lowest_personal": lowest, "wobble": wob} | assess(w, max(0.0, lowest - wob))
    report["strict"] = assess(*STRICT)
    best = max(clients.PERSONAL, key=lambda w: report[w]["tune"]["general_kept"]["k"] + report[w]["grailqa_kept"]["k"])
    return {"personal_wording": best, "personal_bar": report[best]["tune"]["bar"]}, report


def choose_personal(grail):
    """Tunes each wording's bar, then checks every candidate on the held-out businesses. The tuned pick
    replaces the strict baseline only if it leaks no more there, so this check only ever moves stricter.
    Labels reviewed after scoring are counted both ways: only the messages Jev got wrong went to that review."""
    tuned, report = tune_personal(grail)
    held_out, unreviewed = personal_set("test"), personal_set("test", stages=("before scoring",))
    for entry in report.values():
        entry["test"] = judge(entry["wording"], entry["tune"]["bar"], held_out)
        entry["test_without_later_review"] = judge(entry["wording"], entry["tune"]["bar"], unreviewed)
    if report[tuned["personal_wording"]]["test"]["leaks"]["k"] > report["strict"]["test"]["leaks"]["k"]:
        tuned = {"personal_wording": STRICT[0], "personal_bar": STRICT[1]}
    report["chosen"] = tuned
    return tuned, report


def tune_same(pairs):
    """The "same?" bar by the rule, over every pair the checked curve sent to Jev."""
    stored, asked = zip(*pairs)
    wob = wobble(same[1], same[2], stored, asked)
    worst = max((same[1](*pair) for pair, good in pairs.items() if not good), default=0.0)
    report = {"pairs": len(pairs), "wrong_pairs": sum(not g for g in pairs.values()), "worst_wrong": worst,
              "wobble": wob}
    return min(1.0, worst + wob), report


def floor_and_top(alone, checked, timings):
    top = NO_TOP
    for p in reversed(alone):
        if p["wrong_by_path"].get("top"):
            break
        if p["paths"].get("top", 0) >= TOP_HITS:
            top = p["bar"]
    below = [p for p in checked if p["bar"] < top]

    def saved(p):
        return p["paths"].get("checked", 0) * timings["haiku_answer_median_s"] - p["jev"]["checks"] * timings["jev_same_median_s"]

    return {"floor": max(below, key=lambda p: (saved(p), p["bar"]))["bar"] if below else top, "top": top}


def repeat_share(texts):
    """The share of questions whose text appeared earlier in the set: the most an exact match serves."""
    seen, n = set(), 0
    for t in texts:
        n += t in seen
        seen.add(t)
    return n / len(texts)


def main(name):
    start = time.perf_counter()
    rows = read(f"sets/grailqa/set_{name}.csv")
    texts = [r["question"] for r in rows]
    group = {r["question"]: r["group"] for r in rows}
    if len(group) != len({(r["question"], r["group"]) for r in rows}):
        raise ValueError("one question text sits in two groups, so a served answer can't be scored")
    fetch(embed, texts)
    try:
        create_index(connect(BASE), BASE)
    except redis.ResponseError as e:
        if "already exists" not in str(e):
            raise

    if name == "a":
        jev, personal_report = choose_personal(texts)
        Path("results/personal.json").write_text(json.dumps(personal_report, indent=1) + "\n")
        print(f"personal: {jev}", flush=True)
        pairs = {}
        curves(rows, group, name, jev | {"same_bar": BASE.same_bar}, ["checked"], pairs)
        jev["same_bar"], same_report = tune_same(pairs)
        print(f"same: {jev['same_bar']:.3f} from {same_report}", flush=True)
    else:
        bars = json.loads(Path("results/bars.json").read_text())
        jev = {k: bars[k] for k in ("personal_wording", "personal_bar", "same_bar")}
        fetch(personal[jev["personal_wording"], 1], texts)
    pairs = {}
    passes = curves(rows, group, name, jev, ["checked", "alone"], pairs)
    checked, alone = passes[:len(BARS)], passes[len(BARS):]
    if name == "a":
        timings = json.loads(Path("results/timings.json").read_text())
        bars = jev | floor_and_top(alone, checked, timings) | {"set": "a"}
        Path("results/bars.json").write_text(json.dumps(bars, indent=1) + "\n")

    s = replace(BASE, **{k: bars[k] for k in ("floor", "top", "same_bar", "personal_bar")})
    out = run(rows, group, f"{name} gate", s, personal[bars["personal_wording"], 1])
    fam = families(rows, out)
    gate = score(rows, group, out, s) | {"ranges": ranges(fam.values()), "wrong_hits": [
        {"question": r["question"], "family": r["family"], "path": o.path, "check": o.check}
        for r, o in zip(rows, out) if o.served not in (None, r["group"])]}
    general = [r["question"] for r in rows if r["personal"] == "no"]
    p = personal[bars["personal_wording"], 1]
    result = {"set": name, "bars": bars, "gate": gate,
              "layer1": {"raw": repeat_share(texts), "tidy": repeat_share([tidy(t) for t in texts])},
              "general_blocked_by_personal": {"k": sum(p(q) > s.personal_bar for q in general), "n": len(general)},
              "alone": alone, "checked": checked, "families": fam,
              "pairs": sorted([stored, asked, good] for (stored, asked), good in pairs.items())}
    if name == "a":
        result["same_tuning"] = same_report
    Path(f"results/sweep_{name}.json").write_text(json.dumps(result, indent=1) + "\n")

    w, a = gate["ranges"]["wrong_among_hits"], gate["all"]
    print(f"bars: {bars}")
    print(f"gate: hit rate {a['hit_rate']:.1%}, wrong among hits {w['k']}/{w['n']} = {w['rate']:.2%} "
          f"(95% up to {w['high']:.2%}, {w['high_from']}), recall {a['recall']:.1%}, jev {gate['jev']}")
    paid = sum(m.paid for m in [embed, *same.values(), *personal.values()])
    print(f"paid calls: {paid}; {(time.perf_counter() - start) / 60:.1f} min")


def overlap(x, y):
    return x["low"] <= y["high"] and y["low"] <= x["high"]


def verdict():
    """The gate on B and C together. A first run writes the hand-check list: every
    pair from two groups that Jev passed, and a random 100 pairs from one group that it blocked. Once
    each row is marked "Jev wrong" or "label wrong", a second run writes the verdict, as labelled and
    with the wrong labels corrected. The frozen sets are never edited."""
    sweeps = {s: json.loads(Path(f"results/sweep_{s}.json").read_text()) for s in "bc"}
    same_bar = sweeps["b"]["bars"]["same_bar"]
    if not REVIEW.exists():
        seen = [{"set": s, "stored": a, "asked": b, "label": "same" if good else "different",
                 "jev": same[1](a, b), "mark": "", "reason": ""}
                for s, sweep in sweeps.items() for a, b, good in sweep["pairs"]]
        passed = [r for r in seen if r["label"] == "different" and r["jev"] >= same_bar]
        blocked = [r for r in seen if r["label"] == "same" and r["jev"] < same_bar]
        with REVIEW.open("w", encoding="utf-8", newline="") as f:
            out = csv.DictWriter(f, list(seen[0]))
            out.writeheader()
            out.writerows(passed + random.Random(0).sample(blocked, min(REVIEW_SAMPLE, len(blocked))))
        print(f"{REVIEW}: {len(passed)} passed from two groups, {min(REVIEW_SAMPLE, len(blocked))} of "
              f"{len(blocked)} blocked from one group. Mark each row, then run this again.")
        return
    review = read(REVIEW)
    if any(r["mark"] not in ("Jev wrong", "label wrong") for r in review):
        raise ValueError(f"every row of {REVIEW} needs a mark: 'Jev wrong' or 'label wrong'")
    relabelled = {(r["stored"], r["asked"]) for r in review if r["label"] == "different" and r["mark"] == "label wrong"}

    def together(corrected):
        fam = []
        for sweep in sweeps.values():
            fixed = Counter(h["family"] for h in sweep["gate"]["wrong_hits"]
                            if corrected and h["check"] and (h["check"][0], h["question"]) in relabelled)
            fam += [Counter(f) - Counter(wrong=fixed[k]) for k, f in sweep["families"].items()]
        return ranges(fam)

    result = {"rule": "wrong among hits under 1% at the 95% upper bound, sets B and C together",
              "as_labelled": together(False), "corrected": together(True),
              "b_and_c_ranges_overlap": {k: overlap(sweeps["b"]["gate"]["ranges"][k], sweeps["c"]["gate"]["ranges"][k])
                                         for k in sweeps["b"]["gate"]["ranges"]},
              "hand_check": Counter(f"{r['label']}: {r['mark']}" for r in review)}
    for k in ("as_labelled", "corrected"):
        result[k]["passes"] = result[k]["wrong_among_hits"]["high"] < 0.01
    Path("results/gate.json").write_text(json.dumps(result, indent=1) + "\n")
    for k in ("as_labelled", "corrected"):
        w = result[k]["wrong_among_hits"]
        print(f"{k}: wrong among hits {w['k']}/{w['n']} (95% up to {w['high']:.2%}), passes {result[k]['passes']}")
    print(f"B and C ranges overlap: {result['b_and_c_ranges_overlap']}; hand check: {dict(result['hand_check'])}")


if __name__ == "__main__":
    verdict() if sys.argv[1] == "gate" else main(sys.argv[1])
