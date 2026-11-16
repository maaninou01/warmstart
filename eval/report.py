"""Draws the charts and the results page from the files in results/.

  uv run python -m eval.report
"""
import html, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
GATE = 0.01  # the gate: under 1% wrong among served
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "axes.axisbelow": True,
})


def load(name):
    return json.loads((R / f"{name}.json").read_text())


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def pooled(sweeps, mode):
    """Sets B and C summed bar by bar: the 1,777 held-out questions the gate was judged on."""
    rows = []
    for points in zip(*(s[mode] for s in sweeps)):
        assert len({p["bar"] for p in points}) == 1, "the sweeps must share their bars"
        rows.append({"bar": points[0]["bar"], "checks": sum(p["jev"]["checks"] for p in points),
                     **{k: sum(p["all"][k] for p in points) for k in ("questions", "hits", "wrong")}})
    return rows


def save(fig, name):
    fig.tight_layout()
    fig.savefig(R / name, dpi=200)
    plt.close(fig)


def curve_chart(alone, checked, floor, gate):
    """Share of served answers that are right, against share of questions served, one point per bar."""
    fig, ax = plt.subplots(figsize=(8, 5.2))
    # The checked line bunches up between 40% and 48% served, so it is labelled only where it spreads out.
    lines = ((alone, BLUE, "Similarity alone", -13, (0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)),
             (checked, ORANGE, "Similarity + Jev check", 8, (0.9, 0.95, 0.99)))
    for rows, color, name, dy, labelled in lines:
        pts = [(r["bar"], 100 * r["hits"] / r["questions"], 100 * (1 - r["wrong"] / r["hits"])) for r in rows if r["hits"]]
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color=color, lw=2, marker="o", ms=5,
                mec=SURFACE, mew=1, label=name, solid_joinstyle="round")
        for bar, x, y in pts:
            if bar in labelled:
                steep = rows is alone and bar < 0.9
                ax.annotate(f"{bar:.2f}", (x, y), xytext=(-8, -3) if steep else (0, dy), textcoords="offset points",
                            ha="right" if steep else "center", fontsize=8, color=INK2)
    chosen = next(r for r in checked if r["bar"] == floor)
    x, y = 100 * chosen["hits"] / chosen["questions"], 100 * (1 - chosen["wrong"] / chosen["hits"])
    low = 100 * (1 - gate["wrong_among_hits"]["high"])
    ax.errorbar(x, y, yerr=[[y - low], [0]], fmt="o", ms=10, color=ORANGE, mec=INK, mew=1.5, ecolor=INK,
                capsize=4, zorder=5)
    ax.annotate(f"Chosen: floor {floor:.2f}\n{pct(chosen['hits'] / chosen['questions'])} served, "
                f"{chosen['wrong']} wrong in {chosen['hits']}\nbar = 95% range, down to {low:.2f}% right",
                (x, y), xytext=(x - 25, y - 4), fontsize=8.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1, "shrinkB": 8})
    ax.axhline(100 * (1 - GATE), color=MUTED, lw=1, zorder=1)
    ax.text(1, 100 * (1 - GATE) - 0.15, f"Gate: {pct(1 - GATE, 0)} right", fontsize=8, color=INK2, va="top")
    ax.set_xlim(0, None)
    ax.set_xlabel("Questions served from the cache (%)")
    ax.set_ylabel("Served answers that are right (%)")
    ax.set_title(f"Each point is a similarity bar. Sets B and C, {alone[0]['questions']:,} questions")
    ax.legend(loc="lower left")
    save(fig, "precision_curve.png")


def cost_chart(replay):
    names = {"no_cache": "No cache", "all_three_without_prompt_cache": "Exact + semantic, no prompt cache",
             "prompt_cache_and_exact": "Prompt cache + exact match", "prompt_cache": "Prompt cache only",
             "all_three_measured": "All three layers (as run)"}
    costs = sorted(replay["cost_per_1000"].items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh([names[k] for k, _ in costs], [v for _, v in costs], height=0.5, color=BLUE)
    for i, (_, v) in enumerate(costs):
        ax.text(v, i, f"  ${v:.3f}", va="center", fontsize=9, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, costs[-1][1] * 1.15)
    ax.set_xlabel(f"US dollars per 1,000 questions (prices checked {replay['prices']['sources']['checked']})")
    ax.set_title("Cost per 1,000 questions, set B replay")
    save(fig, "cost.png")


def warmup_chart(replay):
    w = replay["warmup"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar([b["questions"] for b in w], [100 * b["hit_rate"] for b in w], width=0.5, color=BLUE)
    for i, b in enumerate(w):
        ax.text(i, 100 * b["hit_rate"] + 1.5, pct(b["hit_rate"], 0), ha="center", fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Questions, in the order asked")
    ax.set_ylabel("Served from the cache (%)")
    ax.set_title("The cache fills as it goes: share served, by question range")
    save(fig, "warmup.png")


def latency_chart(replay):
    lat = replay["latency"]
    rows = [("haiku_alone", "Haiku alone (no cache)"), ("miss", "Miss: cache checks, then Haiku"),
            ("checked", "Served from the cache")]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, (key, _) in enumerate(rows):
        p50, p95, (low, high) = lat[key]["p50"], lat[key]["p95"], lat[key]["p95_range"]
        ax.plot([p50, p95], [i, i], color=GRID, lw=2, zorder=1)
        ax.errorbar(p95, i, xerr=[[p95 - low], [high - p95]], fmt="none", ecolor=MUTED, capsize=3, zorder=2)
        ax.plot(p50, i, "o", ms=9, color=BLUE, mec=SURFACE, mew=2, zorder=3, label="p50 (half are faster)" if i == 0 else None)
        ax.plot(p95, i, "o", ms=9, color=ORANGE, mec=SURFACE, mew=2, zorder=3, label="p95 (19 in 20 are faster)" if i == 0 else None)
        ax.text(p95, i + 0.22, f"{p50:.2f} s / {p95:.2f} s", ha="right", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(rows)), [label for _, label in rows])
    ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, None)
    ax.set_xlabel("Seconds per question (whisker: 95% range of p95)")
    pen = lat["miss_penalty"]
    ax.set_title(f"Answer time by path: a miss adds {pen['p50']:.2f} s (p50)")
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "latency.png")


def overview(gate, replay, personal):
    """The README's top image: the request path as boxes and arrows, then five headline numbers. It carries
    its own light card, so it reads the same on GitHub's light and dark themes."""
    lab, lat, cost = gate["as_labelled"], replay["latency"], replay["cost_per_1000"]
    leaks = personal["strict"]["test"]["leaks"]
    assert personal["chosen"]["personal_bar"] == personal["strict"]["tune"]["bar"], "the leak test must use the shipped bar"
    out = []
    def text(x, y, s, size=12, color=INK2, weight=400, anchor="start", spacing=0):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
                   f'text-anchor="{anchor}" letter-spacing="{spacing}">{html.escape(s)}</text>')
    def rect(x, y, w, h, stroke=GRID, fill="#ffffff"):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    def arrow(x1, y1, x2, y2):
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{INK2}" stroke-width="1.5" marker-end="url(#head)"/>')

    text(32, 46, "Warmstart", 22, INK, 700)
    text(32, 70, "Reuses an earlier answer when a new support question means the same thing: cheaper, faster, and checked.", 14)
    steps = [("Question + scope", "key: model, prompt,", "tools, customer tier"),
             ("1: Exact match", "same text, same scope", "one Redis read"),
             ("2: Meaning match", "nearest stored question", "Jev check: same answer?"),
             ("3: Haiku answers", "system prompt read from", "prompt cache: 1/10 price")]
    xs, w, top, h = [32, 266, 500, 734], 194, 92, 78
    for i, (x, (title, a, b)) in enumerate(zip(xs, steps)):
        rect(x, top, w, h)
        text(x + 16, top + 28, title, 15, INK, 600)
        text(x + 16, top + 49, a)
        text(x + 16, top + 65, b)
        if i:
            arrow(x - 36, top + h / 2, x - 6, top + h / 2)
            if i > 1:
                text(x - 20, top + h / 2 - 7, "miss", 11, MUTED, anchor="middle")
    below = top + h + 36
    for x in xs[1:3]:
        arrow(x + w / 2, top + h + 2, x + w / 2, below - 4)
        text(x + w / 2 + 8, top + h + 23, "hit", 11, MUTED)
    rect(266, below, 428, 48, BLUE, "#eaf2fc")
    text(480, below + 29, f"Served from the cache in {lat['checked']['p50']:.2f} s instead of "
                          f"{lat['haiku_alone']['p50']:.2f} s", 14, INK, 600, "middle")
    arrow(xs[3] + w / 2, top + h + 2, xs[3] + w / 2, below - 4)
    rect(xs[3], below, w, 48, ORANGE, "#fdf0ea")
    text(xs[3] + w / 2, below + 20, "Stored for 24 h, only if", 12, INK, anchor="middle")
    text(xs[3] + w / 2, below + 37, "Jev finds no customer data", 12, INK, anchor="middle")

    p95 = lambda k: lat[k]["p95"]
    wrong = lab["wrong_among_hits"]
    tiles = [("HIT RATE", f"{lab['hit_rate']['rate']:.0%}", "of questions answered", "from the cache"),
             ("COST", f"−{1 - cost['all_three_measured'] / cost['no_cache']:.0%}", "per 1,000 questions",
              f"${cost['no_cache']:.3f} → ${cost['all_three_measured']:.3f}"),
             ("SPEED", f"×{p95('haiku_alone') / p95('checked'):.1f}", "faster when cached",
              f"p95 {p95('haiku_alone'):.2f} s → {p95('checked'):.2f} s"),
             ("PRECISION", f"{100 * (1 - wrong['rate']):.1f}%", "of cached answers right",
              f"{wrong['k']} wrong in {wrong['n']}"),
             ("PRIVACY", f"{leaks['k']} leaks", f"of {leaks['n']} personal test", "questions let through")]
    tw, ty = (896 - 4 * 12) / 5, below + 74
    for i, (label, big, a, b) in enumerate(tiles):
        x = 32 + i * (tw + 12)
        rect(round(x, 1), ty, round(tw, 1), 104)
        text(round(x + 14, 1), ty + 24, label, 11, MUTED, 600, spacing=1)
        text(round(x + 14, 1), ty + 58, big, 28, BLUE, 700)
        text(round(x + 14, 1), ty + 78, a)
        text(round(x + 14, 1), ty + 94, b)
    height = ty + 104 + 20
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}" '
            f'font-family="system-ui, -apple-system, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif" role="img" '
            f'aria-label="Warmstart: the cache layers and its headline results">\n'
            f'<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>\n'
            f'<rect x="0.75" y="0.75" width="958.5" height="{height - 1.5}" rx="14" fill="{SURFACE}" stroke="{GRID}" stroke-width="1.5"/>\n'
            + "\n".join(out) + "\n</svg>\n")


def range_row(label, r, top, line=None):
    """One rate as a 95% range bar on a 0-to-`top` scale, a dot at the measured rate, an optional gate line."""
    at = lambda v: 100 * min(v, top) / top
    gate = f'<i class="gate" style="left:{at(line):.1f}%"></i>' if line else ""
    return (f'<div class="row"><span class="label">{html.escape(label)}</span>'
            f'<span class="track">{gate}<span class="range" style="left:{at(r["low"]):.1f}%;'
            f'width:{at(r["high"]) - at(r["low"]):.1f}%"></span><b style="left:{at(r["rate"]):.1f}%"></b></span>'
            f'<span class="num">{pct(r["rate"], 2)} <small>({pct(r["low"], 2)}–{pct(r["high"], 2)})</small></span></div>')


def page(bars, gate, replay, alone, checked):
    lab, fix, rep = gate["as_labelled"], gate["corrected"], replay["replay"]["ranges"]
    cost, lat = replay["cost_per_1000"], replay["latency"]
    floor_i = next(i for i, r in enumerate(checked) if r["bar"] == bars["floor"])
    data = json.dumps({"alone": alone, "checked": checked})
    verdict = lambda g: "passes" if g["passes"] else "fails"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Warmstart results</title>
<style>
:root {{ --surface:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
  --blue:#2a78d6; --orange:#eb6834; color-scheme:light; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --surface:#1a1a19; --card:#232322;
  --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --blue:#3987e5; --orange:#d95926; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --surface:#1a1a19; --card:#232322; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --blue:#3987e5; --orange:#d95926; color-scheme:dark; }}
body {{ margin:0; background:var(--surface); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:880px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:36px 0 8px; }}
p, small, .note {{ color:var(--ink2); }} small {{ font-size:12px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:12px; margin-top:20px; }}
.tile {{ background:var(--card); border:1px solid var(--grid); border-radius:8px; padding:12px 14px; }}
.tile .v {{ font-size:24px; font-weight:600; }} .tile div:first-child {{ color:var(--ink2); font-size:13px; }}
.row {{ display:grid; grid-template-columns:minmax(120px, 230px) 1fr auto; gap:12px; align-items:center; margin:8px 0; }}
.track {{ position:relative; height:14px; border-bottom:1px solid var(--grid); }}
.range {{ position:absolute; top:5px; height:4px; border-radius:2px; background:var(--blue); opacity:.45; }}
.track b {{ position:absolute; top:2px; width:10px; height:10px; margin-left:-5px; border-radius:50%; background:var(--blue);
  box-shadow:0 0 0 2px var(--surface); }}
.gate {{ position:absolute; top:-4px; bottom:-4px; width:0; border-left:2px solid var(--orange); }}
.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
img {{ width:100%; height:auto; border-radius:8px; border:1px solid var(--grid); background:#fcfcfb; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:right; padding:6px 8px; border-bottom:1px solid var(--grid); }} th:first-child, td:first-child {{ text-align:left; }}
input[type=range] {{ width:100%; accent-color:var(--orange); }}
@media (max-width:560px) {{ .row {{ grid-template-columns:1fr auto; }} .row .track {{ grid-column:1 / -1; grid-row:2; }} }}
</style></head>
<body><main>
<h1>Warmstart: results</h1>
<p>A cache for an online shop's support assistant: it reuses an earlier answer when a new question means the same
thing, and checks before reusing it. Every number here is read from the files in <code>results/</code>.</p>

<div class="tiles">
 <div class="tile"><div>Questions served from the cache</div><div class="v">{pct(rep["hit_rate"]["rate"])}</div>
  <small>live replay, {rep["hit_rate"]["n"]} questions</small></div>
 <div class="tile"><div>Wrong among served</div><div class="v">{lab["wrong_among_hits"]["k"]} in {lab["wrong_among_hits"]["n"]}</div>
  <small>sets B and C, up to {pct(lab["wrong_among_hits"]["high"], 2)} (95% range)</small></div>
 <div class="tile"><div>Cost per 1,000 questions</div><div class="v">${cost["all_three_measured"]:.3f}</div>
  <small>${cost["no_cache"]:.3f} with no cache</small></div>
 <div class="tile"><div>Answer time, p50 / p95</div><div class="v">{lat["all"]["p50"]:.2f} / {lat["all"]["p95"]:.2f} s</div>
  <small>served {lat["checked"]["p50"]:.2f} s, miss {lat["miss"]["p50"]:.2f} s (p50)</small></div>
</div>

<h2>Wrong answers among those served, with their 95% range</h2>
<p class="note">The line is the gate: the top of the range must stay under {pct(GATE, 0)}. "Corrected" re-marks
the pairs that a hand check found mislabelled; it only ever removes errors, so read it as a lower bound.</p>
{range_row(f"Sets B + C, as labelled ({verdict(lab)})", lab["wrong_among_hits"], 2 * GATE, GATE)}
{range_row(f"Sets B + C, corrected ({verdict(fix)})", fix["wrong_among_hits"], 2 * GATE, GATE)}
{range_row("Live replay, set B", rep["wrong_among_hits"], 2 * GATE, GATE)}
<h2>Questions served from the cache, with their 95% range</h2>
{range_row("Sets B + C", lab["hit_rate"], 1)}
{range_row("Live replay, set B", rep["hit_rate"], 1)}

<h2>How many it serves against how many are right</h2>
<p class="note">Lowering the similarity bar serves more questions. Alone, it also serves more wrong answers;
with the Jev check, the share that is right stays flat.</p>
<img src="precision_curve.png" alt="Right answers among those served against questions served, one line for similarity alone and one with the Jev check">
<p><label for="floor">Similarity bar: <b id="bar"></b></label> <small>(chosen: {bars["floor"]:.2f}; below it,
a stored question is never sent to the Jev check)</small></p>
<input id="floor" type="range" min="0" max="{len(checked) - 1}" value="{floor_i}">
<table><thead><tr><th></th><th>Served</th><th>Wrong among served</th><th>Jev checks per 1,000 questions</th></tr></thead>
<tbody><tr id="alone"><td>Similarity alone</td><td></td><td></td><td></td></tr>
<tr id="checked"><td>Similarity + Jev check</td><td></td><td></td><td></td></tr></tbody></table>

<h2>Cost</h2>
<img src="cost.png" alt="Cost per 1,000 questions for each combination of cache layers">
<h2>Warm-up</h2>
<img src="warmup.png" alt="Share of questions served from the cache, per 100 questions, rising as the cache fills">
<h2>Answer time by path</h2>
<img src="latency.png" alt="p50 and p95 answer time for served questions, misses, and Haiku alone">
<p class="note"><small>Prices: {html.escape(" ".join(v for k, v in replay["prices"]["sources"].items() if k != "checked"))}</small></p>
</main>
<script>
const D = {data};
const fmt = (k, n) => n ? (100 * k / n).toFixed(1) + "% (" + k + ")" : "none";
function show() {{
  const i = +document.getElementById("floor").value;
  document.getElementById("bar").textContent = D.checked[i].bar.toFixed(2);
  for (const mode of ["alone", "checked"]) {{
    const r = D[mode][i], cells = document.querySelectorAll("#" + mode + " td");
    cells[1].textContent = fmt(r.hits, r.questions);
    cells[2].textContent = r.hits ? (100 * r.wrong / r.hits).toFixed(2) + "% (" + r.wrong + ")" : "none served";
    cells[3].textContent = Math.round(1000 * r.checks / r.questions);
  }}
}}
document.getElementById("floor").addEventListener("input", show);
show();
</script>
</body></html>
"""


def summary(alone, checked, chosen, replay, timings):
    """The README's derived numbers: similarity alone set to match the chosen setting on share served,
    then on share wrong; and the hit rate above which the semantic layer saves money. Its extra cost per
    question is the "personal?" check, the Titan vector and the "same?" checks; each hit saves one answer.
    A single "same?" check pays for itself when more than its share of an answer's time, or cost, of the
    checked pairs become hits."""
    wrong = lambda r: r["wrong"] / r["hits"]
    served = [r for r in alone if r["hits"]]
    calls, questions = replay["per_call"], sum(replay["paths"].values())
    extra = sum(calls[s]["mean_cost"] * calls[s]["calls"] for s in ("personal", "titan", "same")) / questions
    answer = {"with_prompt_cache": calls["haiku"]["mean_cost"], "without_prompt_cache": replay["cost_per_1000"]["no_cache"] / 1000}
    return {"sets": "b and c, summed bar by bar", "curves": {"alone": alone, "checked": checked}, "chosen": chosen,
            "alone_same_share_served": min(served, key=lambda r: abs(r["hits"] - chosen["hits"])),
            "alone_same_share_wrong": max((r for r in served if wrong(r) <= wrong(chosen)), key=lambda r: r["hits"]),
            "semantic_layer": {"extra_cost_per_question": extra, "answer_cost": answer,
                               "break_even_hit_rate": {k: extra / v for k, v in answer.items()}},
            "same_check_break_even": {"time": timings["ratio"], "money": calls["same"]["mean_cost"] / calls["haiku"]["mean_cost"]}}


def main():
    bars, gate, replay, timings = load("bars"), load("gate"), load("replay"), load("timings")
    sweeps = [load(f"sweep_{s}") for s in "bc"]
    alone, checked = pooled(sweeps, "alone"), pooled(sweeps, "checked")
    chosen = next(r for r in checked if r["bar"] == bars["floor"])
    assert (chosen["hits"], chosen["wrong"]) == (gate["as_labelled"]["wrong_among_hits"]["n"],
                                                 gate["as_labelled"]["wrong_among_hits"]["k"]), "the curve must meet the gate"
    curve_chart(alone, checked, bars["floor"], gate["as_labelled"])
    cost_chart(replay)
    warmup_chart(replay)
    latency_chart(replay)
    (R / "overview.svg").write_text(overview(gate, replay, load("personal")), encoding="utf-8")
    (R / "index.html").write_text(page(bars, gate, replay, alone, checked), encoding="utf-8")
    (R / "summary.json").write_text(json.dumps(summary(alone, checked, chosen, replay, timings), indent=1) + "\n")
    print("wrote", ", ".join(sorted(p.name for p in R.glob("*.png"))), "index.html and summary.json")


if __name__ == "__main__":
    main()
