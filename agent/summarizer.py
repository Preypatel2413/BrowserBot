# summarizer.py
"""
LLM summarization pipeline for the BrowserBot agent.

Two-stage summarization:
1) summarize_pages(pages): summarize each scraped page into compact bullets
2) synthesize_answer(query, page_summaries): combine summaries into final answer

Requirements:
- pip install openai python-dotenv
- .env contains OPENAI_API_KEY=...

Input format expected (from websearch_tool.py):
pages: List[dict] where each dict contains:
  - rank: int
  - title: str
  - url: str
  - text: str
  - word_count: int
  - content_hash: str
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI


# -----------------------------
# Data models
# -----------------------------

@dataclass
class PageSummary:
    rank: int
    title: str
    url: str
    summary_bullets: List[str]
    key_takeaways: List[str]
    notes: List[str]


@dataclass
class FinalAnswer:
    query: str
    answer: str
    sources: List[Dict[str, str]]  # [{title, url}]


# -----------------------------
# Summarizer
# -----------------------------

class Summarizer:
    """
    Uses OpenAI to:
    - create per-page summaries
    - synthesize a final answer from those summaries

    Defaults are intentionally conservative for reliability.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_page_chars: int = 12_000,
    ):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_page_chars = max_page_chars

    def summarize_pages(self, pages: List[Dict[str, Any]]) -> List[PageSummary]:
        """
        Summarize each page independently.
        Returns structured summaries (bullets + takeaways + notes).
        """
        summaries: List[PageSummary] = []
        for p in pages:
            text = (p.get("text") or "").strip()
            if not text:
                continue

            # keep page text bounded for prompt size
            if len(text) > self.max_page_chars:
                text = text[: self.max_page_chars]

            rank = int(p.get("rank", 0))
            title = str(p.get("title", "")).strip()
            url = str(p.get("url", "")).strip()

            ps = self._summarize_single_page(rank=rank, title=title, url=url, text=text)
            if ps:
                summaries.append(ps)

        return summaries

    def synthesize_answer(self, query: str, page_summaries: List[PageSummary]) -> FinalAnswer:
        """
        Combine page summaries into a single final answer.
        Output: final answer + sources list.
        """
        query = query.strip()
        if not query:
            raise ValueError("query is empty")

        if not page_summaries:
            return FinalAnswer(
                query=query,
                answer="I couldn't find enough useful information from the scraped pages to answer this query.",
                sources=[],
            )

        # Prepare compact input for synthesis
        summaries_payload = []
        for s in page_summaries:
            summaries_payload.append({
                "rank": s.rank,
                "title": s.title,
                "url": s.url,
                "bullets": s.summary_bullets,
                "takeaways": s.key_takeaways,
                "notes": s.notes,
            })

        system = (
            "You are a web research assistant. You will be given a user query and structured summaries "
            "from multiple web pages. Produce a helpful final answer that directly addresses the user query, "
            "preferring overlap/consensus across sources. Avoid hallucinating facts not present in the summaries. "
            "If sources disagree, mention the disagreement briefly. Keep the answer concise but useful."
        )

        user = {
            "query": query,
            "page_summaries": summaries_payload,
            "instructions": {
                "format": (
                    "Return a final answer in Markdown with:\n"
                    "1) A short direct answer (2-4 sentences)\n"
                    "2) A bulleted list of key points (5-10 bullets)\n"
                    "3) A 'Sources' section listing 3-5 most relevant URLs\n"
                ),
                "do_not": [
                    "Do not invent URLs",
                    "Do not claim you visited pages beyond the provided summaries",
                ],
            },
        }

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=self.temperature,
        )

        answer_text = resp.output_text.strip()

        # pick top 5 sources from summaries (by rank)
        sources = [{"title": s.title or f"Result {s.rank}", "url": s.url} for s in sorted(page_summaries, key=lambda x: x.rank)]
        sources = sources[:5]

        return FinalAnswer(query=query, answer=answer_text, sources=sources)

    # -----------------------------
    # Internal prompts
    # -----------------------------

    def _summarize_single_page(self, rank: int, title: str, url: str, text: str) -> Optional[PageSummary]:
        system = (
            "You are a web page summarizer. Summarize the content into compact bullets. "
            "Focus on information relevant to likely user questions. "
            "Remove boilerplate and navigation text. Do not invent facts."
        )

        # Request JSON output so it is easy to parse and stable.
        user = {
            "page": {"rank": rank, "title": title, "url": url},
            "content": text,
            "output_schema": {
                "summary_bullets": "list[str] (5-8 bullets, each <= 20 words)",
                "key_takeaways": "list[str] (2-4 items, each <= 18 words)",
                "notes": "list[str] (0-3 items; e.g., credibility, bias, date, limitations)",
            },
        }

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=self.temperature,
        )

        raw = resp.output_text.strip()

        # Parse JSON safely (LLM should comply, but handle fallback)
        try:
            data = json.loads(raw)
        except Exception:
            # Fallback: make a minimal summary without failing the pipeline
            bullets = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip()]
            bullets = bullets[:8]
            return PageSummary(
                rank=rank,
                title=title,
                url=url,
                summary_bullets=bullets,
                key_takeaways=[],
                notes=["Model did not return strict JSON; used fallback parsing."],
            )

        bullets = data.get("summary_bullets") or []
        takeaways = data.get("key_takeaways") or []
        notes = data.get("notes") or []

        # Defensive shaping
        bullets = [str(b).strip() for b in bullets if str(b).strip()][:8]
        takeaways = [str(t).strip() for t in takeaways if str(t).strip()][:4]
        notes = [str(n).strip() for n in notes if str(n).strip()][:3]

        return PageSummary(
            rank=rank,
            title=title,
            url=url,
            summary_bullets=bullets,
            key_takeaways=takeaways,
            notes=notes,
        )


# -----------------------------
# CLI smoke test
# -----------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str)
    ap.add_argument("--websearch-json", type=str, required=True, help="Path to JSON output from websearch_tool.py")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    args = ap.parse_args()

    with open(args.websearch_json, "r", encoding="utf-8") as f:
        ws = json.load(f)

    pages = ws.get("results", [])
    s = Summarizer(model=args.model)

    page_summaries = s.summarize_pages(pages)
    final = s.synthesize_answer(args.query, page_summaries)

    print(json.dumps({
        "page_summaries": [asdict(x) for x in page_summaries],
        "final": asdict(final)
    }, indent=2, ensure_ascii=False))
