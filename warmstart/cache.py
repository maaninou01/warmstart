"""The answer cache: an exact match, then a semantic match, behind a write gate.

The four services are plain functions:
  embed(question)                 -> list of numbers (the question's vector)
  same(stored_question, question) -> probability that one reply answers both
  personal(question)              -> probability that the answer needs one customer's records
  answer(question, scope)         -> the reply text
"""
import hashlib, json, logging, struct, unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import redis
from redis.backoff import NoBackoff
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.retry import Retry

log = logging.getLogger(__name__)
REDIS_DOWN = (redis.ConnectionError, redis.TimeoutError)


@dataclass(frozen=True)
class Settings:
    floor: float  # similarity below this is a miss, with no check
    top: float  # similarity at or above this serves with no check; above 1 means no top zone
    same_bar: float = 0.50  # "same?" must score at least this to serve
    personal_bar: float = 0.20  # "personal?" must score at most this to serve or to store
    ttl: int = 86_400  # seconds an entry lives
    scope_parts: tuple = ("model", "prompt", "tools", "tier")  # everything that changes the answer
    index: str = "warmstart"
    prefix: str = "warmstart:"
    dim: int = 1024
    host: str = "127.0.0.1"  # not "localhost", which tries IPv6 first
    port: int = 6380
    redis_timeout: float = 0.25


def tidy(text):
    """NFKC, lowercase, single spaces. Numbers and words like "not" are kept."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def connect(s):
    """No retries: with redis-py's default ones, a dead Redis cost 15.8 s per question."""
    return redis.Redis(host=s.host, port=s.port, socket_timeout=s.redis_timeout,
                       socket_connect_timeout=s.redis_timeout, retry=Retry(NoBackoff(), 0),
                       decode_responses=True)


def create_index(r, s):
    """One exact (FLAT) cosine index over every entry under the prefix."""
    vec = VectorField("vec", "FLAT", {"TYPE": "FLOAT32", "DIM": s.dim, "DISTANCE_METRIC": "COSINE"})
    r.ft(s.index).create_index([TagField("ns"), vec],
                               definition=IndexDefinition(prefix=[s.prefix], index_type=IndexType.HASH))


class Cache:
    def __init__(self, settings, embed, same, personal, answer):
        self.s, self.embed, self.same, self.personal, self.answer = settings, embed, same, personal, answer
        self.r = connect(settings)
        self.pool = ThreadPoolExecutor(max_workers=8)

    def ask(self, question, scope):
        """Returns (answer, path). The path is exact, top, checked or miss, or the failure that
        made it answer without the cache: redis_error, embed_error, same_error, personal_error."""
        s, ns = self.s, self._namespace(scope)
        key = f"{s.prefix}{ns}:{digest(tidy(question))}"
        try:
            stored = self.r.hget(key, "answer")
        except REDIS_DOWN:
            return self._without_cache(question, scope, "redis_error")
        if stored is not None:
            return stored, "exact"

        personal = self.pool.submit(self._personal, question)  # runs while the question is embedded

        def general():
            p = personal.result()
            return p is not None and p <= s.personal_bar

        try:
            vec = self.embed(question)
        except Exception:
            return self._without_cache(question, scope, "embed_error")
        if len(vec) != s.dim:
            raise ValueError(f"the embedder returned {len(vec)} numbers, the index holds {s.dim}")
        vec = struct.pack(f"<{s.dim}f", *vec)
        query = (Query(f"@ns:{{{ns}}}=>[KNN 1 @vec $vec AS dist]")
                 .return_fields("question", "answer", "dist").dialect(2))
        try:
            docs = self.r.ft(s.index).search(query, query_params={"vec": vec}).docs
        except REDIS_DOWN:
            return self._without_cache(question, scope, "redis_error")

        if docs:
            near, similarity = docs[0], 1 - float(docs[0].dist)
            if similarity >= s.top:
                return near.answer, "top"
            if similarity >= s.floor:
                try:
                    same = self.same(near.question, question)
                except Exception:
                    return self._without_cache(question, scope, "same_error")
                if same >= s.same_bar and general():
                    return near.answer, "checked"

        reply = self.answer(question, scope)
        if personal.result() is None:
            return reply, "personal_error"
        if general():  # the write gate: only general answers are ever stored
            entry = {"ns": ns, "vec": vec, "question": question, "answer": reply}
            try:
                self.r.pipeline().hset(key, mapping=entry).expire(key, s.ttl).execute()
            except REDIS_DOWN:
                log.warning("redis_error: the answer was not stored", exc_info=True)
                return reply, "redis_error"
        return reply, "miss"

    def invalidate(self, scope):
        """Deletes every entry in this scope's namespace, and nothing else."""
        keys = list(self.r.scan_iter(f"{self.s.prefix}{self._namespace(scope)}:*", count=1000))
        if keys:
            self.r.delete(*keys)

    def _namespace(self, scope):
        """One cache per scope: changing any part, even one word of the prompt, opens a fresh one."""
        if sorted(scope) != sorted(self.s.scope_parts):
            raise ValueError(f"the scope must hold exactly {self.s.scope_parts}, got {tuple(scope)}")
        return digest(json.dumps(scope, sort_keys=True))

    def _personal(self, question):
        """The probability, or None when the check failed. A failed check counts as personal."""
        try:
            return self.personal(question)
        except Exception:
            log.warning("personal_error: the question counts as personal", exc_info=True)
            return None

    def _without_cache(self, question, scope, path):
        log.warning("%s: answering without the cache", path, exc_info=True)
        return self.answer(question, scope), path
