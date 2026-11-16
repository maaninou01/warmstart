# Warmstart evaluation sets

Three sets of about 900 questions each, used to test a semantic answer cache. Each question has a group: questions in one group ask the same thing, so one stored answer serves them all. The cache never sees the group. Only the scorer does.

| Set | Role | Questions | Groups | Expected hits | Expected misses |
|---|---|---|---|---|---|
| `set_a.csv` | tune the similarity bars | 931 | 386 | 545 | 41% |
| `set_b.csv` | test | 874 | 371 | 503 | 42% |
| `set_c.csv` | test | 903 | 382 | 521 | 42% |

## Columns

| Column | Meaning | Who reads it |
|---|---|---|
| `qid` | GrailQA question id | scorer |
| `question` | the question text | **the cache** (the only column it sees) |
| `group` | GrailQA logical form (`s_expression`). Same logical form, same answer | scorer |
| `family` | the template (first 7 digits of `qid`). Groups in one family differ by one entity, number or date | scorer |
| `kind` | `repeat` (a group of 5), `near_miss` (two groups of 2 from one family), `singleton` | scorer |
| `personal` | `no` for every row: GrailQA asks general-knowledge questions | scorer |
| `expected` | `miss` for the first question of its group in file order, `hit` for every later one | scorer |

**The file order is the replay order.** Another company's data plugs in by filling `question`, `group`, `family` (the group itself when no groups are near-misses of each other), `kind` (any label; results are split by it) and `personal`; `expected` follows from the order.

## Source and licence

[GrailQA](https://dki-lab.github.io/GrailQA/) (Gu et al., 2021), train and validation splits, 51,100 crowd-written questions about the Freebase knowledge graph. Licence: CC BY-SA 4.0. These derived sets carry the same licence.

## How it was built

1. **`build_sets.py build`** (seed 42) draws three sets with no template in common.
   - Each set holds 120 groups of 5 questions, 60 families × 2 groups × 2 questions, and 160 singletons.
   - Near-miss siblings differ only by an entity, a number or a date: "episode 870" vs "episode 1354"; "aired after 2005-10-17" vs "after 2007-01-09". These are the hard negatives.
2. **`label_check.py`** asks Claude Haiku 5.5 (Bedrock, eu-west-1), once per question, whether each grouped question asks exactly what its logical form retrieves.
   - Haiku sees the logical form with readable names and a sample of its answers.
   - It runs in two prompt versions. Version 2 adds one rule, written after reading version 1's misses.
3. **`build_sets.py freeze`** drops every question that either version flagged, then adds `personal` and `expected`.

Singletons aren't checked: a question with no partner can't make a pair wrong.

## How far to trust the groups

**About 6–7% of the grouped questions didn't ask what their logical form retrieves**, against the 2.1% the GrailQA paper reports. The check flagged 292 of 2,520 (11.6%). A review of 30 random flags found 17 real errors, 9 false alarms and 4 unsure, which puts the real rate at 6.6% (95% range 4.5–8.5%). Every flagged question was dropped. Typical cases:

- **A relation read backwards:** "which writing system does the Phoenician alphabet fall under?", where the logical form returns the systems that come from it.
- **Another class:** "who colored chapter seven?", where the logical form returns the inker.
- **A constraint the logical form doesn't have:** "…in any of Ken Akamatsu's works".

**Why this matters:** a wrong question left in a group biases the result in the dangerous direction. If the cache serves it, the scorer marks a false hit as correct.

**How many errors the check misses** was measured on planted errors. A question is paired with a real GrailQA logical form that it does *not* match, and the check must reject it. Only questions that match their own logical form are used.

| Planted error | Caught by either version | 95% interval |
|---|---|---|
| A sibling's logical form (another entity or value) | 581 of 587 (99.0%) | 97.8–99.5% |
| Same entity, another relation, direction or class, drawn fresh (neither version was written against it) | 462 of 476 (97.1%) | 95.1–98.2% |

**Remaining errors:** with about 165 real errors caught at a 95–98% catch rate, roughly 3–10 errors remain among the 2,228 grouped questions kept. That's under 0.5%. Three caveats:
- Planted errors may be easier than real ones, so this is a best estimate, not a guarantee.
- **The checker flags some good questions too:** 9 of 30 flags (95% range 17–48%), so about 90 good questions were dropped. That costs data, not correctness.
- **The 30 flags were reviewed by a second model** (Claude Opus; `label_check/flag_review.csv`, column `opus_verdict`), not by a person. It isn't fully independent of the checker.

**Duplicate groups (the other direction):** two groups asking the same thing would make a correct hit look false. Two equivalent groups must return identical answers.
- 132 pairs of groups in the same set share an answer list.
- Only 1 of those pairs names the same entity, and it asks different questions ("tallest guitarist" vs "heaviest guitarist").
- Same answer is not same question: "International System of Units" answers about 20 different questions, one per unit.

## Known limits

- **Recall looks better than it would on real traffic.** Each group's rewordings were written from one seed question.
- **Some questions read oddly** ("dat is the icao designer for which airline?"), because crowd workers wrote them from Freebase's own terms.
- **Dropping flagged questions may remove hard but correct rewordings.** That makes recall look better again. It doesn't affect the false-hit rate.
- **There are no personal questions.** The personal check's accuracy isn't measured on this data; its evidence is in `DECISIONS.md`.

## Regenerate

```
uv run python -I eval/build_sets.py build data/grailqa sets/grailqa/label_check
uv run python -I eval/label_check.py data/grailqa sets/grailqa/label_check 1 5000
uv run python -I eval/label_check.py data/grailqa sets/grailqa/label_check 2 5000
uv run python -I eval/build_sets.py freeze sets/grailqa/label_check sets/grailqa
```

Haiku's verdicts are saved in `label_check/label_check_v1.jsonl` and `label_check_v2.jsonl`, so `freeze` reruns without paying again. Cost of the full check: $2.13.
