"""Checks that each question asks exactly what its group's logical form retrieves.

  uv run python -I eval/label_check.py <grailqa_dir> <sets_dir> <version> <limit>
"""
import csv, json, random, re, sys, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from anthropic import AnthropicBedrock

MODEL = "eu.anthropic.claude-haiku-5-5"
# USD per million tokens: list price x 1.1 for a regional profile; the eu. price is assumed equal to the us. one.
PRICE_IN, PRICE_OUT = 0.11, 0.55
ENTITY = re.compile(r"\b[mg]\.[0-9a-z_]+\b")  # Freebase ids such as m.0j59b

SYSTEM = """You audit a question-answering dataset built on the Freebase knowledge graph. A crowd worker wrote each question to ask exactly what a given logical form retrieves. You see the question, the logical form, readable names for its Freebase ids, its function and its answer.

How to read a logical form: (JOIN r X) is everything whose relation r points to X; (R r) is r read backwards; (AND A B) keeps what is in both; COUNT counts the results; (ARGMAX A v) and (ARGMIN A v) keep the item of A with the largest or smallest value v, where v is a relation or a path written (JOIN r1 r2): follow r1, then read r2, so (ARGMAX brand (JOIN brand.models model.discontinued)) is the brand whose model was discontinued last; gt, ge, lt and le compare a value (greater, greater or equal, less, less or equal). The answer list is what the logical form really returns: use it to confirm which side of a relation is being asked for.

Decide whether the question asks for exactly what the logical form retrieves.

It is a mismatch when the question changes what is asked: {kind}another entity, value, date or number; another relation, or the relation read in the wrong direction; a constraint added or dropped; a count asked as a list, or a list as a count; a comparison or superlative changed (more or less, before or after, most or least).

It is a match when only the wording differs: synonyms, word order, grammar mistakes, awkward phrasing, capitalisation, or another reasonable name for the same entity or class."""

KIND_RULE = ("the logical form returns another kind of thing than the question asks for (its first "
             "class: a city when the question asks for a country, a genre when it asks for a subject); ")
PROMPTS = {1: SYSTEM.replace("{kind}", ""), 2: SYSTEM.replace("{kind}", KIND_RULE)}
HARD_SEEDS = {"plant_hard": 43, "plant_fresh": 44}

SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "match": {"type": "boolean"}},
    "required": ["reason", "match"],
    "additionalProperties": False,
}

client = AnthropicBedrock(aws_region="eu-west-1", timeout=60.0, max_retries=2)
lock = threading.Lock()


def meaning(row):
    """The logical form in words Haiku can read: readable names, function and a few answers."""
    nodes, edges, answer = row["graph_query"]["nodes"], row["graph_query"]["edges"], row["answer"]
    names = [f'{i} = "{n}"' for i, n in zip(nodes["id"], nodes["friendly_name"])]
    names += [f'{r} = "{n}"' for r, n in zip(edges["relation"], edges["friendly_name"])]
    args = answer["answer_argument"]
    shown = [n or a for n, a in zip(answer["entity_name"] or [None] * len(args), args)]
    return (f"logical form: {row['s_expression']}\n"
            f"names: {'; '.join(names)}\n"
            f"function: {row['function']}\n"
            f"answer ({len(shown)} items, first 5 shown): {'; '.join(map(str, shown[:5]))}")


def load(src, sets_dir):
    cols = ["qid", "s_expression", "graph_query", "answer", "function"]
    info, by_template, by_entity = {}, defaultdict(set), defaultdict(set)
    for f in sorted(Path(src).glob("*.parquet")):
        for r in pq.read_table(f, columns=cols).to_pylist():
            info.setdefault(r["s_expression"], r)
            by_template[r["qid"][:7]].add(r["s_expression"])
            for e in ENTITY.findall(r["s_expression"]):
                by_entity[e].add(r["s_expression"])
    rows = []
    for f in sorted(Path(sets_dir).glob("set_*.csv")):
        with open(f, encoding="utf-8") as fh:
            rows += [r for r in csv.DictReader(fh) if r["kind"] != "singleton"]
    return info, by_template, by_entity, rows


def plan(info, by_template, by_entity, rows):
    rng = random.Random(42)
    hard = {kind: random.Random(seed) for kind, seed in HARD_SEEDS.items()}
    tasks = []
    for r in rows:
        task = {"qid": r["qid"], "question": r["question"]}
        tasks.append({**task, "kind": "real", "group": r["group"]})
        siblings = sorted(by_template[r["family"]] - {r["group"]})
        if siblings:
            tasks.append({**task, "kind": "plant", "group": rng.choice(siblings)})
        same_entity = sorted(s for e in ENTITY.findall(r["group"]) for s in by_entity[e]
                             if info[s]["qid"][:7] != r["family"])
        if same_entity:
            tasks += [{**task, "kind": kind, "group": hard_rng.choice(same_entity)}
                      for kind, hard_rng in hard.items()]
    rng.shuffle(tasks)
    return tasks


def check(task, info, version):
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=PROMPTS[version],
        messages=[{"role": "user", "content":
                   f"<question>{task['question']}</question>\n\n"
                   f"<logical_form>\n{meaning(info[task['group']])}\n</logical_form>"}],
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"stopped on {response.stop_reason}")
    reply = json.loads(next(b.text for b in response.content if b.type == "text"))
    return {"qid": task["qid"], "kind": task["kind"], "group": task["group"],
            "match": reply["match"], "reason": reply["reason"],
            "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return (p + z * z / (2 * n) - half) / d, (p + z * z / (2 * n) + half) / d


def summary(out):
    rows = [json.loads(line) for line in open(out, encoding="utf-8")] if out.exists() else []
    usd = sum(r["input_tokens"] * PRICE_IN + r["output_tokens"] * PRICE_OUT for r in rows) / 1e6
    for kind, verb in (("real", "flagged"), ("plant", "caught"), ("plant_hard", "caught"),
                       ("plant_fresh", "caught")):
        done = [r for r in rows if r["kind"] == kind]
        k = sum(not r["match"] for r in done)
        lo, hi = wilson(k, len(done))
        print(f"{kind}: {k} of {len(done)} {verb} ({100 * k / max(len(done), 1):.1f}%, "
              f"95% interval {100 * lo:.1f}-{100 * hi:.1f}%)")
    print(f"${usd:.3f} so far")


def main(src, sets_dir, version, limit):
    info, by_template, by_entity, rows = load(src, sets_dir)
    tasks = plan(info, by_template, by_entity, rows)
    out = Path(sets_dir) / f"label_check_v{version}.jsonl"
    done = {(r["qid"], r["kind"]) for r in map(json.loads, open(out, encoding="utf-8"))} if out.exists() else set()
    todo = [t for t in tasks if (t["qid"], t["kind"]) not in done][:limit]
    print(f"{len(tasks)} checks planned, {len(done)} saved, {len(todo)} to run now")
    errors = 0

    def run(task):
        nonlocal errors
        try:
            row = check(task, info, version)
        except Exception as e:  # left unsaved, so the next run retries it
            with lock:
                errors += 1
                print(f"error on {task['qid']} {task['kind']}: {e}")
            return
        with lock, open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(run, todo))
    print(f"{errors} errors")
    summary(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
