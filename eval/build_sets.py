"""Builds three disjoint 1,000-question evaluation sets from GrailQA, then freezes them.

  uv run python -I eval/build_sets.py build <grailqa_dir> <check_dir>
  uv run python -I eval/build_sets.py freeze <check_dir> <out_dir>
"""
import csv, json, random, re, sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

SETS = ("a", "b", "c")
REPEAT, REPEAT_SIZE, NEAR_MISS, SINGLETONS = 120, 5, 60, 160


def load(src):
    rows = []
    for f in sorted(Path(src).glob("*.parquet")):
        rows += pq.read_table(f, columns=["qid", "question", "s_expression"]).to_pylist()
    groups = defaultdict(list)
    for r in rows:
        groups[r["s_expression"]].append(r)
    families = defaultdict(list)
    for s, members in groups.items():
        fams = {r["qid"][:7] for r in members}
        if len(fams) == 1:  # 3 groups span two templates; dropped so the family split stays clean
            families[fams.pop()].append(s)
    return groups, families


def build(groups, families, rng):
    big_fams = {f for f, ss in families.items() if any(len(groups[s]) >= REPEAT_SIZE for s in ss)}
    repeat = sorted(s for f in big_fams for s in families[f] if len(groups[s]) >= REPEAT_SIZE)
    near = sorted(f for f, ss in families.items()
                  if f not in big_fams and sum(len(groups[s]) >= 2 for s in ss) >= 2)
    lone = sorted(f for f in families if f not in big_fams and f not in near
                  and any(len(groups[s]) == 1 for s in families[f]))
    for pool in (repeat, near, lone):
        rng.shuffle(pool)
    need = len(SETS)
    assert len(repeat) >= need * REPEAT and len(near) >= need * NEAR_MISS and len(lone) >= need * SINGLETONS

    out = {}
    for i, name in enumerate(SETS):
        rows = []
        for s in repeat[i * REPEAT:(i + 1) * REPEAT]:
            rows += [(r, "repeat") for r in rng.sample(groups[s], REPEAT_SIZE)]
        for f in near[i * NEAR_MISS:(i + 1) * NEAR_MISS]:
            pair = rng.sample(sorted(s for s in families[f] if len(groups[s]) >= 2), 2)
            rows += [(r, "near_miss") for s in pair for r in rng.sample(groups[s], 2)]
        for f in lone[i * SINGLETONS:(i + 1) * SINGLETONS]:
            s = rng.choice(sorted(s for s in families[f] if len(groups[s]) == 1))
            rows.append((groups[s][0], "singleton"))
        rng.shuffle(rows)
        out[name] = rows
    return out


def check(sets):
    norm = lambda t: re.sub(r"\s+", " ", t.lower()).strip()
    seen_fams = set()
    for name, rows in sets.items():
        assert len(rows) == 1000, name
        fams = {r["qid"][:7] for r, _ in rows}
        assert not fams & seen_fams, "a template appears in two sets"
        seen_fams |= fams
        text_group = {}
        for r, _ in rows:
            assert text_group.setdefault(norm(r["question"]), r["s_expression"]) == r["s_expression"]
        n_groups = len({r["s_expression"] for r, _ in rows})
        hits = len(rows) - n_groups
        near = sum(k == "near_miss" for _, k in rows)
        print(f"set_{name}: 1000 questions, {n_groups} groups, {hits} should-hit "
              f"({100 - hits / 10:.0f}% should-miss), {near} near-miss questions")


def write(src, dst):
    groups, families = load(src)
    sets = build(groups, families, random.Random(42))
    check(sets)
    for name, rows in sets.items():
        with open(Path(dst) / f"set_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["qid", "question", "group", "family", "kind"])
            w.writerows([r["qid"], r["question"], r["s_expression"], r["qid"][:7], k] for r, k in rows)


def freeze(check_dir, out_dir):
    """Drop every question the label check flagged, and add each question's expected results.

    The expected columns are for the scorer only; the cache never sees them:
      personal  "no": every GrailQA question is general knowledge
      expected  "miss" for the first question of its group in file order (the replay order),
                "hit" for every later one
    """
    check_dir, out_dir = Path(check_dir), Path(out_dir)
    versions = sorted(check_dir.glob("label_check_v*.jsonl"))
    assert versions, "run label_check.py first"
    flagged, checked = set(), []
    for path in versions:
        verdicts = [json.loads(line) for line in open(path, encoding="utf-8")]
        checked.append({v["qid"] for v in verdicts if v["kind"] == "real"})
        flagged |= {v["qid"] for v in verdicts if v["kind"] == "real" and not v["match"]}
    checked = set.intersection(*checked)
    out_dir.mkdir(exist_ok=True)
    for f in sorted(check_dir.glob("set_*.csv")):
        with open(f, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows, columns = list(reader), [*reader.fieldnames, "personal", "expected"]
        unchecked = [r["qid"] for r in rows if r["kind"] != "singleton" and r["qid"] not in checked]
        assert not unchecked, f"{f.name}: {len(unchecked)} grouped questions have no verdict yet"
        kept, seen = [r for r in rows if r["qid"] not in flagged], set()
        for r in kept:
            r["personal"] = "no"
            r["expected"] = "hit" if r["group"] in seen else "miss"
            seen.add(r["group"])
        with open(out_dir / f.name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=columns)
            w.writeheader()
            w.writerows(kept)
        hits = sum(r["expected"] == "hit" for r in kept)
        print(f"{f.name}: {len(rows) - len(kept)} dropped, {len(kept)} kept, {len(seen)} groups, "
              f"{hits} expected hits ({100 - 100 * hits / len(kept):.0f}% expected misses)")


if __name__ == "__main__":
    {"build": write, "freeze": freeze}[sys.argv[1]](*sys.argv[2:])
