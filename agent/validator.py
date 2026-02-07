from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


_DEFAULT_MODEL = "gpt-4o-mini"
_ENV_PATH = ".env"

load_dotenv(_ENV_PATH)


def is_web_search_feasible(query: str, *, client: Optional[OpenAI] = None, model: Optional[str] = None) -> bool:
    """Return True if the query is feasible using browser-based search and scraping."""
    if not query or not query.strip():
        return False

    api_key = os.environ.get("OPENAI_API_KEY")
    # print(api_key)
    if not api_key and client is None:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = client or OpenAI(api_key=api_key)
    model = model or os.environ.get("BROWSERBOT_VALIDATOR_MODEL", _DEFAULT_MODEL)


    system_prompt = (
        "You are a validator that answers if a user query can be handled by browser-based "
        "searching and web scraping. Reply with only 'YES' or 'NO'."
    )
    user_prompt = (
        "Query:\n"
        f"{query}\n\n"
        "Is this query feasible to answer using browser-based searching and scraping? "
        "Answer ONLY YES or NO."
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        # max_output_tokens=3,
    )

    text = response.output_text.strip().upper()
    return text.startswith("YES")


def _run_smoke_tests() -> None:
    queries = [
        "What is the latest NVIDIA stock price?",
        "Write a short poem about rain.",
        "Who is the CEO of OpenAI?",
        "Solve 2+2.",
        "Find the official documentation for FastAPI.",
    ]

    for query in queries:
        try:
            result = is_web_search_feasible(query)
        except Exception as exc:
            print(f"Query: {query}\nError: {exc}\n")
            continue
        print(f"Query: {query}\nFeasible: {'YES' if result else 'NO'}\n")


if __name__ == "__main__":
    _run_smoke_tests()
