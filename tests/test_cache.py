"""Attacks and unhappy paths for the cache engine, on a real Redis with fake services."""
import time
from dataclasses import replace

import pytest

from warmstart.cache import Cache, Settings, connect, create_index

TEST = Settings(floor=0.80, top=0.95, index="warmstart_test", prefix="warmstart_test:", dim=4)
SCOPE = {"model": "haiku", "prompt": "You answer shop questions.", "tools": [], "tier": "basic"}
BASE = [1.0, 0.0, 0.0, 0.0]


def near(similarity):
    """A vector whose cosine similarity with BASE is exactly `similarity`."""
    return [similarity, (1 - similarity**2) ** 0.5, 0.0, 0.0]


class Fake:
    """The four services, scripted per question. A scripted exception is raised, not returned."""

    def __init__(self):
        self.vectors, self.personal_p, self.same_p, self.calls = {}, {}, 1.0, []

    def _reply(self, name, args, value):
        self.calls.append((name, *args))
        if isinstance(value, Exception):
            raise value
        return value

    def embed(self, q):
        return self._reply("embed", (q,), self.vectors.get(q, BASE))

    def same(self, a, b):
        return self._reply("same", (a, b), self.same_p)

    def personal(self, q):
        return self._reply("personal", (q,), self.personal_p.get(q, 0.0))

    def answer(self, q, scope):
        n = sum(c[0] == "answer" for c in self.calls) + 1
        return self._reply("answer", (q,), f"answer {n} to {q}")

    def called(self, name):
        return [c[1:] for c in self.calls if c[0] == name]


def build(**changes):
    fake = Fake()
    return Cache(replace(TEST, **changes), fake.embed, fake.same, fake.personal, fake.answer), fake


def stored():
    return len(connect(TEST).keys(TEST.prefix + "*"))


@pytest.fixture(autouse=True)
def index():
    r = connect(TEST)
    create_index(r, TEST)
    yield
    r.ft(TEST.index).dropindex(delete_documents=True)


def test_exact_match_ignores_case_and_spacing():
    cache, _ = build()
    cache.ask("What is the return window?", SCOPE)
    assert cache.ask("  what IS the   return window? ", SCOPE) == ("answer 1 to What is the return window?", "exact")


def test_zones_decide_who_gets_checked():
    cache, fake = build()
    cache.ask("stored", SCOPE)
    fake.vectors.update(top=near(0.99), middle=near(0.90), low=near(0.50))
    assert cache.ask("top", SCOPE) == ("answer 1 to stored", "top")
    assert cache.ask("low", SCOPE) == ("answer 2 to low", "miss")
    assert fake.called("same") == []
    assert cache.ask("middle", SCOPE) == ("answer 1 to stored", "checked")
    assert fake.called("same") == [("stored", "middle")]


def test_personal_answer_is_never_stored():
    cache, fake = build()
    fake.personal_p["Where is my order 4417?"] = 0.9
    cache.ask("Where is my order 4417?", SCOPE)
    assert stored() == 0
    fake.personal_p.clear()  # the next customer's identical message is judged general
    assert cache.ask("Where is my order 4417?", SCOPE) == ("answer 2 to Where is my order 4417?", "miss")


@pytest.mark.parametrize("p, path", [(0.5, "miss"), (RuntimeError("Jev is down"), "personal_error")])
def test_unsure_or_failed_personal_counts_as_personal(p, path):
    cache, fake = build()
    cache.ask("How do refunds work?", SCOPE)
    fake.vectors["How does my refund work?"] = near(0.90)  # the middle zone, and "same?" passes
    fake.personal_p["How does my refund work?"] = p
    assert cache.ask("How does my refund work?", SCOPE) == ("answer 2 to How does my refund work?", path)
    assert stored() == 1


@pytest.mark.parametrize("part", TEST.scope_parts)
def test_every_scope_part_isolates(part):
    cache, _ = build()
    cache.ask("What is the return window?", SCOPE)
    assert cache.ask("What is the return window?", {**SCOPE, part: "other"})[1] == "miss"
    assert cache.ask("What is the return window?", SCOPE)[1] == "exact"


def test_prompt_edit_busts_the_cache():
    cache, _ = build()
    cache.ask("What is the return window?", SCOPE)
    edited = {**SCOPE, "prompt": SCOPE["prompt"] + " Be brief."}
    assert cache.ask("What is the return window?", edited)[1] == "miss"


def test_failed_same_never_serves():
    cache, fake = build()
    cache.ask("How do refunds work?", SCOPE)
    fake.vectors["How does a refund work?"] = near(0.90)
    fake.same_p = TimeoutError("Jev timed out")
    assert cache.ask("How does a refund work?", SCOPE) == ("answer 2 to How does a refund work?", "same_error")
    assert stored() == 1


@pytest.mark.parametrize("down", ["redis", "embed"])
def test_still_answers_when_a_service_is_down(down):
    cache, fake = build(port=1) if down == "redis" else build()  # nothing listens on port 1
    fake.vectors["Do you ship abroad?"] = ConnectionError("Titan is down")
    start = time.perf_counter()
    assert cache.ask("Do you ship abroad?", SCOPE) == ("answer 1 to Do you ship abroad?", f"{down}_error")
    assert time.perf_counter() - start < 1


def test_entries_expire():
    cache, _ = build(ttl=1)
    cache.ask("What is the return window?", SCOPE)
    assert cache.ask("What is the return window?", SCOPE)[1] == "exact"
    time.sleep(1.2)
    assert cache.ask("What is the return window?", SCOPE)[1] == "miss"


def test_wipe_empties_its_namespace_only():
    cache, _ = build()
    pro = {**SCOPE, "tier": "pro"}
    cache.ask("What is the return window?", SCOPE)
    cache.ask("What is the return window?", pro)
    cache.invalidate(SCOPE)
    assert cache.ask("What is the return window?", SCOPE)[1] == "miss"
    assert cache.ask("What is the return window?", pro)[1] == "exact"
