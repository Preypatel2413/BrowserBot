# similarity.py
"""
Similarity-based query cache using SQLite + sqlite-vec.

What it does
- store(query, response): embeds query with OpenAI, stores response + embedding
- lookup(query): embeds query, finds top_k nearest stored embeddings using sqlite-vec,
  then computes cosine similarity in Python and returns the stored response if
  best_score >= threshold.

Requirements
- pip install sqlite-vec openai
- export OPENAI_API_KEY="..."

Notes
- This implementation stores *unit-normalized* float32 embeddings (for cosine sim).
- It uses sqlite-vec only to shortlist candidates (top_k), then does the final
  cosine similarity check in Python so the "score >= threshold" behavior is stable.
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
from dataclasses import dataclass
from typing import Optional, Tuple, List

from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI  # modern SDK

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = (BASE_DIR.parent / "data" / "similarity_cache.sqlite")



# --------- OpenAI embeddings (modern SDK only) ----------

class _Embedder:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Query text is empty; cannot embed.")

        resp = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )
        return resp.data[0].embedding


# --------- sqlite-vec helpers ----------

def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    try:
        import sqlite_vec  # type: ignore
        sqlite_vec.load(conn)
    finally:
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass


def _serialize_f32(vec: List[float]) -> bytes:
    """
    Convert list[float] -> compact float32 BLOB.
    Uses sqlite_vec.serialize_float32 if available; else uses struct.pack.
    """
    try:
        from sqlite_vec import serialize_float32  # type: ignore
        return serialize_float32(vec)
    except Exception:
        return struct.pack("<" + "f" * len(vec), *[float(x) for x in vec])


def _deserialize_f32(blob: bytes, dim: int) -> List[float]:
    expected = dim * 4
    if len(blob) != expected:
        raise ValueError(f"Unexpected embedding blob length: {len(blob)} (expected {expected})")
    return list(struct.unpack("<" + "f" * dim, blob))


def _l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _unit_normalize(vec: List[float]) -> List[float]:
    n = _l2_norm(vec)
    if n == 0.0:
        return vec[:]
    return [x / n for x in vec]


def _cosine_similarity_unit(a_unit: List[float], b_unit: List[float]) -> float:
    return sum(x * y for x, y in zip(a_unit, b_unit))


# --------- Cache implementation ----------

@dataclass
class MatchInfo:
    hit: bool
    score: float
    matched_id: Optional[int]
    matched_query: Optional[str]


class SimilarityCache:
    def __init__(
        self,
        db_path: str = DEFAULT_DB,
        embedding_model: str = "text-embedding-3-small",
        dim: Optional[int] = None,
        threshold: float = 0.8,
        top_k: int = 5,
    ):
        if not (-1.0 <= threshold <= 1.0):
            raise ValueError("threshold must be between -1 and 1.")
        if top_k <= 0:
            raise ValueError("top_k must be >= 1.")

        self.db_path = db_path
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.top_k = top_k

        self._embedder = _Embedder(model=embedding_model)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        _load_sqlite_vec(self.conn)

        self._init_schema()
        self.dim = self._init_or_check_meta(dim)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS cache_items (
                id INTEGER PRIMARY KEY,
                query_text TEXT NOT NULL,
                response_text TEXT NOT NULL
            );
        """)

        self.conn.commit()

    def _init_or_check_meta(self, dim: Optional[int]) -> int:
        cur = self.conn.cursor()

        model_row = cur.execute("SELECT value FROM meta WHERE key='embedding_model'").fetchone()
        dim_row = cur.execute("SELECT value FROM meta WHERE key='embedding_dim'").fetchone()

        if model_row is None and dim_row is None:
            # Fresh DB
            if dim is None:
                inferred = len(self._embedder.embed("dimension probe"))
            else:
                inferred = dim

            self._set_meta_and_create_vec(model=self.embedding_model, dim=inferred)
            return inferred

        existing_model = model_row[0]
        existing_dim = int(dim_row[0])

        if existing_model != self.embedding_model:
            raise ValueError(
                f"DB was created with embedding_model={existing_model}, "
                f"but you requested {self.embedding_model}. Use a new DB or rebuild."
            )

        if dim is not None and existing_dim != dim:
            raise ValueError(
                f"DB was created with dim={existing_dim}, but you requested dim={dim}. "
                "Use a new DB or rebuild."
            )

        self._ensure_vec_table(existing_dim)
        return existing_dim

    def _set_meta_and_create_vec(self, model: str, dim: int) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('embedding_model', ?)", (model,))
        cur.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('embedding_dim', ?)", (str(dim),))
        self._ensure_vec_table(dim)
        self.conn.commit()

    def _ensure_vec_table(self, dim: int) -> None:
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS cache_vec USING vec0(
                embedding float[{dim}]
            );
        """)

    def _embed_unit(self, text: str) -> List[float]:
        emb = self._embedder.embed(text)
        if len(emb) != self.dim:
            raise ValueError(f"Embedding dim mismatch: got {len(emb)}, expected {self.dim}")
        return _unit_normalize([float(x) for x in emb])

    def store(self, query: str, response: str) -> int:
        q = query.strip()
        if not q:
            raise ValueError("query is empty")
        if response is None:
            raise ValueError("response is None")

        vec_unit = self._embed_unit(q)
        vec_blob = _serialize_f32(vec_unit)

        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO cache_items(query_text, response_text) VALUES(?, ?)",
            (q, response),
        )
        item_id = int(cur.lastrowid)

        cur.execute(
            "INSERT INTO cache_vec(rowid, embedding) VALUES(?, ?)",
            (item_id, vec_blob),
        )

        self.conn.commit()
        return item_id

    def lookup(self, query: str) -> Tuple[bool, Optional[str], MatchInfo]:
        q = query.strip()
        if not q:
            return False, None, MatchInfo(hit=False, score=float("-inf"), matched_id=None, matched_query=None)

        vec_unit = self._embed_unit(q)
        vec_blob = _serialize_f32(vec_unit)

        # IMPORTANT: vec0 KNN requires "k = ?" constraint (not LIMIT ?)
        rows = self.conn.execute(
            """
            SELECT
            nn.id AS id,
            nn.distance AS distance,
            nn.embedding_blob AS embedding_blob,
            c.query_text AS query_text,
            c.response_text AS response_text
            FROM (
            SELECT
                rowid AS id,
                distance AS distance,
                embedding AS embedding_blob
            FROM cache_vec
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            ) AS nn
            JOIN cache_items c ON c.id = nn.id;
            """,
            (vec_blob, self.top_k),
        ).fetchall()

        if not rows:
            return False, None, MatchInfo(hit=False, score=float("-inf"), matched_id=None, matched_query=None)

        best_score = float("-inf")
        best_resp: Optional[str] = None
        best_id: Optional[int] = None
        best_query: Optional[str] = None

        for rid, _dist, emb_blob, q_text, resp_text in rows:
            try:
                cand_unit = _deserialize_f32(emb_blob, self.dim)
            except Exception:
                continue

            score = _cosine_similarity_unit(vec_unit, cand_unit)
            if score > best_score:
                best_score = score
                best_resp = resp_text
                best_id = int(rid)
                best_query = q_text

        hit = best_resp is not None and best_score >= self.threshold
        info = MatchInfo(hit=hit, score=best_score, matched_id=best_id, matched_query=best_query)
        return hit, (best_resp if hit else None), info


# --------- tiny CLI demo ----------

def _demo():
    import argparse

    p = argparse.ArgumentParser(description="SimilarityCache demo")
    p.add_argument("query", type=str, help="Query text to lookup")
    p.add_argument("--db", type=str, default=DEFAULT_DB)
    p.add_argument("--model", type=str, default="text-embedding-3-small")
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--store-response", type=str, default=None, help="If provided, store this response for the query.")
    args = p.parse_args()

    cache = SimilarityCache(
        db_path=args.db,
        embedding_model=args.model,
        threshold=args.threshold,
        top_k=args.topk,
    )

    if args.store_response is not None:
        rid = cache.store(args.query, args.store_response)
        print(json.dumps({"stored": True, "id": rid}, indent=2))
        return

    hit, resp, info = cache.lookup(args.query)
    print(json.dumps({
        "hit": hit,
        "response": resp,
        "score": info.score,
        "matched_id": info.matched_id,
        "matched_query": info.matched_query,
        "threshold": args.threshold,
        "top_k": args.topk,
    }, indent=2))


if __name__ == "__main__":
    _demo()
