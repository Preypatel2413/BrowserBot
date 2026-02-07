# BrowserBot

## Setup
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies.
   ```powershell
   pip install -r BrowserBot/requirements.txt
   ```
3. Install the Playwright browser.
   ```powershell
   python -m playwright install chromium
   ```
4. Create `BrowserBot/agent/.env` and set your key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

Run the app:
```
python BrowserBot/main.py
```

## High Level Design

<img width="942" height="1262" alt="Workflow" src="https://github.com/user-attachments/assets/a2fa655d-b112-48ca-b617-9a8a4517aec4" />

BrowserBot is an LLM‑assisted web research agent. The flow is:
1. Validate the user query with an LLM to decide if web search is appropriate.
2. Check the similarity cache for a close match and return cached response if found.
3. Rewrite the user query into a concise search query.
4. Search and scrape top results using a browser tool.
5. Summarize each page and synthesize a final answer.
6. Store the final answer in the similarity cache.

Key components:
- `agent.py`: Orchestrates the end‑to‑end flow.
- `validator.py`: LLM‑based “YES/NO” feasibility check.
- `websearch_tool.py`: Playwright search + scraping.
- `summarizer.py`: LLM summaries and final synthesis.
- `similarity.py`: SQLite + sqlite‑vec cache for near‑duplicate queries.

## Details
Environment variables:
- `OPENAI_API_KEY`: required for validator, summarizer, and embeddings.

Similarity cache:
- DB path: `BrowserBot/data/similarity_cache.sqlite`
- Uses sqlite‑vec to run KNN search over embeddings.
- Stores normalized, unit‑length embeddings for cosine similarity.

File map:
- `BrowserBot/main.py`: CLI entry point.
- `BrowserBot/agent/agent.py`: agent orchestration.
- `BrowserBot/agent/validator.py`: LLM feasibility check.
- `BrowserBot/agent/websearch_tool.py`: web search + scrape.
- `BrowserBot/agent/summarizer.py`: page summaries + final answer.
- `BrowserBot/agent/similarity.py`: embedding cache + vector search.
