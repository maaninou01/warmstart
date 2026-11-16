# Decisions

Each row gives a decision, why it was made, what it beat, and when to revisit it on another company's data. "First design" marks a rule that was changed during the build, and why.

| Decision | Why | Rejected | Revisit when |
|---|---|---|---|
| **Data: GrailQA,** in three frozen sets. A sets the bars; B and C only report. | Its questions come in labelled groups of rewordings, plus near-miss families that differ by one detail. Those are exactly the pairs a cache must refuse. | ABCD: 52 answer groups, about 2% should-miss questions, and about 10% of its labels looked wrong. | You have labelled traffic of your own. |
| **Check the labels before measuring.** Drop every question that either of two Haiku prompts flags ($2.13). | A wrong label inside a group scores a false hit as right, which is the dangerous direction. | Trusting the paper's 2.1%: the real rate was about 6.6%. | Never skip it. |
| **Never edit a frozen set.** Hand-check findings are reported twice, as labelled and corrected. | A set changed after its results are seen is no longer a fair test. | Fixing labels in place. | — |
| **The key is the scope:** model, prompt hash, tool set and tier, hashed into a namespace. Versions are content hashes. | It holds everything that changes the answer and nothing else, since each extra part splits the cache. Any edit opens a fresh namespace, so invalidation can't be forgotten. | A key on the question alone, which serves one tier's or prompt's answer to another. Hand-bumped version numbers. | Something else changes the answer: add it as a part. |
| **Write gate:** store an answer only if "personal?" calls it general. | No hit can carry one customer's data to another. | Caching personal answers per customer: the same customer rarely asks twice, and their data changes in between. | Never, for a shared cache. |
| **Serve the stored answer word for word,** with no check on answers before storing. | A wrong answer was already wrong for the first customer. | An answer judge. | Poisoning becomes a real threat. Until then, `invalidate(scope)` is the remedy. |
| **Jev for both yes/no checks,** pinned to `jev-1.14.0`: one call per check, no retries, a 2 s timeout. Errors and unsure scores block. | It takes about 0.17 s and costs about a quarter of a Haiku answer (evidence below). | Haiku as the checker. Averaging two calls, which ranked no better. An LLM second opinion when Jev is unsure: Haiku was right only 45–70% of the time there. | A new Jev version: re-tune before moving the pin. |
| **"Same?" bar 0.542:** the highest-scoring wrong pair on A (0.41), plus one wobble (0.132). The wobble is how far a second identical call moves a score, at the 99th percentile. | It leaves a margin for the score's own noise, and it's set on A only. | First design: start at 0.50 and move stricter only after wrong passes on A. That ignored the noise. | — |
| **"Personal?" bar 0.20, wording V2,** the strict baseline. A tuned bar replaces it only if it leaks no more on held-out businesses. | The tuned bar (0.23) let 1 of 505 held-out personal messages through; the strict one let none. | The tuned bar. Wording V3, which blocked every general message. | Your own personal messages: re-tune on them first. |
| **Floor 0.66:** on A, the bar that saves the most time, counted as hits × answer time − checks × Jev time. | Time binds first. A "same?" check pays for its wait when more than 11% of checked pairs become hits, and for its cost above 26%. | First design: the last 0.01 step before the hit share drops under 11%. It ignored how many checks each step adds. | Your answer model is much faster or cheaper than Haiku. |
| **No top zone,** so similarity alone never serves without a check. | No bar on A served the 381 hits with none wrong needed to bound the error under 1% (Wilson). | A top bar picked by eye. | A bar on your data clears 381 hits with none wrong. |
| **Every 95% range spans both methods:** a Wilson interval and a family bootstrap, from the lower low to the higher high. | Neither method alone can make a rate look safer. The wider range can still top out lower and pass a gate that Wilson fails (`eval/metrics.py` holds a case). | First design: report the wider of the two ranges. | — |
| **The gate:** under 1% wrong among served, at the top of the 95% range, on B and C together, with the bars frozen on A first. | It is the target, judged at its worst plausible value, on questions the bars never saw. | Judging the point estimate. | — |
| **Prompt cache:** one marker after the system prompt, with the 5-minute cache. The tier goes after the marker. | At 400k questions a month (about 9 a minute) it never goes cold, and a 1-hour cache write costs 2× the input price instead of 1.25×. | The 1-hour cache. | A scope sees fewer than one question every 5 minutes. |
| **Titan V2 embeddings (1,024 numbers) in Redis 8.10.2,** with an exact (FLAT) index. | Titan runs on demand in eu-west-1 at $0.026 per million tokens (the first design assumed $0.02). Vector search is built into Redis, so evaluation and production run the same code. | A second embedding model. Fine-tuning, which needs labelled pairs at scale. | You're choosing between vendors, or thousands of labelled pairs exist. |
| **Fail open:** if Redis or Titan is down, answer without the cache. The Redis client uses no retries, a 0.25 s timeout and `127.0.0.1`. | With redis-py's default retries, a dead Redis cost 15.8 s per question, and `localhost` tries IPv6 first. | Failing the question. | — |

The time and cost break-even shares come from `results/summary.json`. The prices are dated in `results/replay.json`.

## Evidence for the Jev checks

- The "personal?" check, measured on held-out businesses (`results/personal.json`).
- Jev's agreement with the GrailQA labels (`results/gate.json`).
- **Known weak spots:**
  - personal questions that never say "my" ("How many minutes are left this month?");
  - very short messages ("refund?");
  - two questions one word apart, where only the policy tells them apart.

## Left out on purpose

| Left out | Why | Worth it when |
|---|---|---|
| A bar per stored entry (as in vCache) | One global bar is enough to show the trade-off. | The traffic mixes very different kinds of question. |
| Caching follow-up messages | Rewriting each one costs an LLM call, and there's no data to score it. | Follow-ups are a large share of traffic. |
| Template answers | It's a different project. | Answers are mostly fill-in-the-blank. |
| A production service, live counters, load tests | It's a different project. | Shipping it. |
| Defences against crafted collisions | Stated as a known limit. | The cache is public-facing. |
| Wiping by tag, or a time limit per category | It needs dependency tags or categories first. | Policies change often. |
| An LLM judge at scoring time | The frozen labels do the job. | Live traffic with no labels. |
| Free rules in front of Jev (place names must match; "my order" never reuses) | Jev's two checks cover both. | Jev is unavailable. |
| A cost model per layer and traffic mix | The break-even formulas cover it. | Budgeting a rollout. |
