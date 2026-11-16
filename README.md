# Warmstart

An answer cache for a shop's AI support assistant, which answers 400,000 questions a month. Most of them are the
same few questions in other words ("where is my order?", "how do returns work?"). Warmstart reuses an earlier answer
when a new question means the same thing, so the shop pays the model less and the customer waits less.

Building the cache is the easy part: Redis, an embedding model, and the provider's prompt cache. The hard part is
proving it never gives a customer a wrong answer, someone else's data, or an answer from before the last prompt
change. A cache that does any of those is worse than no cache at all.

![Warmstart: a question goes through exact match, then meaning match with a yes/no check, then Haiku with a prompt cache. 48% are served from the cache, cost per 1,000 questions falls 77%, a cached answer is 5.3 times faster at p95, 99.6% of cached answers are right, and 0 of 505 personal questions leaked.](results/overview.svg)

Measured on 1,777 questions its settings never saw, then replayed live through the real services: Claude Haiku on
AWS Bedrock, Titan embeddings, and a small yes/no model (TypeSafe Jev) as the checker.

## Design decisions

- **The cache key holds everything that changes the answer.** The model, the system prompt, the tool set and the
  customer's tier are hashed into the key, so a question only meets answers given under the same conditions. A key
  on the question alone would serve one tier's answer to another. Editing one word of the prompt starts a fresh,
  empty cache by itself, so there is no version number to forget.
- **No personal answer is stored.** Before an answer is saved, Jev checks whether it depends on one customer's
  data, such as an order or an address. If it does, it is never stored, so no later hit can carry it to someone
  else.
- **Similar wording isn't the same question.** "Episode 870" and "episode 1354" look almost identical to an
  embedding model but need different answers. So similarity only finds a candidate, and Jev then checks that one
  reply answers both. Serving the same 48% of questions, similarity alone got 16 wrong in 848; with the check, 3 in
  853.
- **How wrong the cache may be is a business choice, made before the test.** Every threshold comes from a written
  rule applied to one set of questions, then frozen; the results come from two other sets. The similarity floor sits
  where a check still saves time: it pays once more than 11% of checked questions become hits.
- **The provider's prompt cache does most of the saving.** Reading the long system prompt from Bedrock's prompt
  cache costs a tenth of the normal price and cuts 1,000 questions from $0.291 to $0.073. The meaning cache only
  takes it to $0.068, because it adds an embedding and a privacy check to every question, and a "same?" check to
  about half. It saves money only above a 40%
  hit rate (10% without prompt caching). Its real gain is speed.
- **Error bars that can't flatter.** Each result's 95% range covers two standard methods, and the 1% target is
  judged at the top of that range, not at the average.
- **A broken cache never breaks the answer.** If Redis or the embedding service is down, the assistant answers
  without the cache, at most 0.25 s later. If the privacy check fails, the question is treated as personal.

## Proof that it works

- **3 wrong answers in 853 served (0.35%).** It served 48% of 1,777 held-out questions. The target was under 1%: at
  the top of its 95% range (1.03%) it misses by 0.03 points as labelled. A hand check found 2 of the 3 were label
  errors, which brings the top to 0.66%, but that check could only lower the count, so 0.35% is the figure to quote.
- **No personal answer reached another customer.** None of 505 personal test questions got through (up to 0.75%).
- **It holds up live.** One set was replayed through the real services, one question at a time: 46.3% served, and
  every token count matched AWS's own. A cached answer takes 0.28 s instead of 1.44 s.
- **A prompt change empties the cache.** The tests change each part of the key in turn and expect a miss, and try to
  make one customer's answer reach another.
- **The trade-off is on one page.** [`results/index.html`](results/index.html) (download it; it works offline) shows
  right answers against questions served at every similarity threshold, with a slider.

## What it covers, and what other goals would add

- **Built for the typical wait.** A cached answer comes back 5.3 times faster. The slowest 5% of all answers are new
  questions, so the p95 across all traffic moves from 2.32 s to 2.20 s, and a miss adds 0.12 s. A goal set on the
  slowest answers would add streaming, so the customer sees the first words sooner.
- **Measured on a public benchmark with labelled rewordings (GrailQA).** Every hit can be scored right or wrong,
  which is what a precision number needs. A shop with its own traffic would label a sample into groups and rerun the
  tuning on it.
- **The privacy check is strict on purpose.** It was tuned on shop messages written for 8 fictional businesses, and
  it holds back a third of general questions to let no personal one through. A shop would re-tune it on its own
  messages to keep more.
- **Stored answers are trusted for 24 hours.** A stored answer is served until it expires or its scope is wiped. A
  public-facing assistant would add a check on answers before storing them.
- **One live replay, one question at a time.** That measures the cost and wait per question. A production rollout
  would add a load test against a deployed service.

## Reproduce

You need Docker and [uv](https://docs.astral.sh/uv/). The paid steps also need AWS Bedrock access in eu-west-1 and a
`.env` copied from [`.env.example`](.env.example).

```
docker run -d --name warmstart-redis --restart unless-stopped -p 127.0.0.1:6380:6379 redis:8.10.2
uv sync
uv run pytest                                     # the engine, leak attacks and failure paths (free)
uv run python -m eval.report                       # the image, charts and results page from results/ (free)
uv run --env-file .env python -m eval.sweep a      # sets and freezes the thresholds on A; then b, c and gate
uv run --env-file .env python -m eval.replay run   # the live replay of set B (about 6 cents)
uv run --env-file .env python -m eval.replay report
```

`warmstart/` is the cache itself. `eval/` is how every number was made, `sets/` the frozen questions, `results/`
every output, and `memo/` the saved Jev scores that make reruns free.

Every number here comes from a file in [`results/`](results/). Each decision, what it beat and when to revisit it on
other data is in [DECISIONS.md](DECISIONS.md). To run it on your own questions, start from the
[datasheet](sets/grailqa/DATASHEET.md).

The code is under the MIT licence. The question sets in `sets/` come from GrailQA and carry its CC BY-SA 4.0 licence.
