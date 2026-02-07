from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from agent.similarity import SimilarityCache
from agent.summarizer import Summarizer
from agent.validator import is_web_search_feasible
from agent.websearch_tool import WebSearchTool


_DEFAULT_SEARCH_MODEL = "gpt-4o-mini"


def _load_env() -> None:
    agent_env = Path(__file__).resolve().parent / ".env"
    project_env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(agent_env)
    load_dotenv(project_env)


def _build_search_query(query: str, *, client: OpenAI, model: Optional[str] = None) -> str:
    model = model or os.environ.get("BROWSERBOT_SEARCH_MODEL", _DEFAULT_SEARCH_MODEL)
    system = (
        "You rewrite user questions into concise web search queries. "
        "Make sure to not change meaning of user's query."
        "Return only the query text with no quotes, no bullets, and no extra words."
    )
    user = (
        "User question:\n"
        f"{query}\n\n"
        "Rewrite as a concise search query."
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_output_tokens=32,
    )
    search_query = resp.output_text.strip()
    return search_query or query


def run_agent(query: str) -> str:
    """Main orchestration entrypoint for BrowserBot."""
    query = query.strip()
    if not query:
        return "This is not a valid query."

    _load_env()

    try:
        if not is_web_search_feasible(query):
            return "This query is not feasible with my capabilities. Please let me know what else I can do for you."
    except Exception as exc:
        return f"Validation failed: {exc}"

    cache: Optional[SimilarityCache] = None
    try:
        cache = SimilarityCache()
        hit, cached_resp, _info = cache.lookup(query)
        if hit and cached_resp:
            cache.close()
            return cached_resp
    except Exception:
        cache = None

    client = OpenAI()
    try:
        search_query = _build_search_query(query, client=client)
    except Exception:
        search_query = query

    engine = os.environ.get("BROWSERBOT_SEARCH_ENGINE", "duckduckgo")
    tool = WebSearchTool(top_k=5, engine=engine, headless=False, pause_on_captcha=False)
    try:
        search_result = tool.search_and_scrape(search_query)
    except Exception as exc:
        return f"Web search failed: {exc}"

    pages = search_result.get("results", [])
    if not pages:
        return "I couldn't find useful results for this query."

    summarizer_model = os.environ.get("BROWSERBOT_SUMMARIZER_MODEL", "gpt-4o-mini")
    summarizer = Summarizer(model=summarizer_model)
    page_summaries = summarizer.summarize_pages(pages)
    final = summarizer.synthesize_answer(query, page_summaries)

    answer = (final.answer or "").strip()
    if not answer:
        return "I couldn't generate a useful answer from the scraped pages."

    if cache is not None:
        try:
            cache.store(query, answer)
        except Exception:
            pass
        finally:
            cache.close()

    return answer
