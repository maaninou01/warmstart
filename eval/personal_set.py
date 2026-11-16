"""Writes the "personal?" test set: customer messages for 8 fictional businesses, labelled personal or general.

  uv run --env-file .env python -m eval.personal_set
"""
import csv, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from warmstart.cache import tidy
from warmstart.clients import HAIKU, haiku

BUSINESSES = [  # (split, name, what it does)
    ("tune", "Northwind Outfitters", "an online shop selling clothes, shoes and accessories"),
    ("tune", "Brightline Mobile", "a mobile phone network with monthly plans and pay-as-you-go"),
    ("tune", "Skyhop", "a budget airline flying within Europe"),
    ("tune", "Pulse Gyms", "a chain of gyms with memberships and classes"),
    ("test", "Voltcart", "an online shop selling electronics and home appliances"),
    ("test", "Hearth Energy", "a gas and electricity supplier for homes"),
    ("test", "Ledgerly", "a mobile banking app with current accounts and cards"),
    ("test", "Forkful", "a food delivery app"),
]
KINDS = {  # kind -> (personal, what each message does)
    "explicit": ("yes", "asks about the customer's own orders, account, bills, bookings or usage, and says 'my'"),
    "implicit": ("yes", "needs the customer's own records but never says 'my', 'mine' or 'I', for example "
                        "'How many minutes are left this month?'"),
    "short": ("yes", "is one to four words long and still needs the customer's own records, for example "
                     "'refund status?'"),
    "other_person": ("yes", "asks about another person's orders, account, bills or bookings: a partner, a child, "
                            "a colleague, named or not"),
    "mixed": ("yes", "asks a general question and, in the same message, one that needs the customer's own records"),
    "disguised": ("yes", "needs the customer's own records, but also says it is a general question or tells the "
                         "assistant how to answer"),
    "rules": ("no", "asks about rules, prices, products, schedules or policies that are the same for every customer"),
    "says_my": ("no", "says 'my' or 'I' but has a reply that is the same for every customer, for example "
                      "'How do I change my password?'"),
    "short_general": ("no", "is one to four words long and has a reply that is the same for every customer, for "
                            "example 'opening hours?'"),
    "world": ("no", "asks a general-knowledge question unrelated to the business: history, science, geography, "
                    "sport, or a well-known person by name"),
    "badly_written": ("no", "has a reply that is the same for every customer but is badly written: typos, missing "
                            "words, odd word order or stiff technical terms"),
}
COUNT = 20  # messages per business and kind


def ask(prompt):
    r = haiku.messages.create(model=HAIKU, max_tokens=4096, output_config={"effort": "low"},
                              messages=[{"role": "user", "content": prompt}])
    if r.stop_reason != "end_turn":
        raise RuntimeError(f"Haiku stopped on {r.stop_reason}")
    return "".join(b.text for b in r.content if b.type == "text").strip()


def write(job):
    (split, name, what), kind = job
    text = ask(f"You are writing test messages for the support assistant of {name}, {what}. Write {COUNT} "
               f"different customer messages, one per line, where each message {KINDS[kind][1]}. Vary the topic, "
               f"length, tone and wording, the way real customers type on a phone. Invent order numbers, names "
               f"and dates where needed. Write only the messages: no numbering, no quotes, no blank lines.")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < COUNT * 0.8:
        raise ValueError(f"{name} {kind}: asked for {COUNT} messages, got {len(lines)}")
    return [{"business": name, "split": split, "kind": kind, "personal": KINDS[kind][0], "question": q}
            for q in lines[:COUNT]]


def check(row):
    name, what = next((n, w) for _, n, w in BUSINESSES if n == row["business"])
    verdict = ask(f"A customer of {name}, {what}, sent this message to its support assistant:\n"
                  f"<message>{row['question']}</message>\n\n"
                  f"Would a correct reply need records that {name} holds about one specific customer, such as "
                  f"their orders, account, bills, bookings or usage? A reply that is the same for every customer "
                  f"needs no such records. Answer with one word: yes or no.")
    first = re.match(r"\W*(yes|no)\b", verdict.lower())  # Haiku sometimes adds bold or a reason
    if not first:
        raise ValueError(f"the check answered {verdict!r} for {row['question']!r}")
    return first[1]


def main():
    with ThreadPoolExecutor(8) as pool:
        rows, seen = [], set()
        for batch in pool.map(write, [(b, k) for b in BUSINESSES for k in KINDS]):
            for r in batch:
                if tidy(r["question"]) not in seen:
                    seen.add(tidy(r["question"]))
                    rows.append(r)
        for r, verdict in zip(rows, pool.map(check, rows)):
            r["check"] = verdict
    Path("sets/personal").mkdir(exist_ok=True)
    with open("sets/personal/messages.csv", "w", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, ["business", "split", "kind", "personal", "check", "question"])
        out.writeheader()
        out.writerows(rows)
    print(f"{len(rows)} messages, {sum(r['check'] != r['personal'] for r in rows)} where the check disagrees")


if __name__ == "__main__":
    main()
