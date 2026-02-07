from __future__ import annotations

import re
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# -----------------------------
# Data models
# -----------------------------

@dataclass
class ScrapedPage:
    rank: int
    title: str
    url: str
    text: str
    word_count: int
    content_hash: str


@dataclass
class WebSearchResult:
    query: str
    engine: str
    results: List[ScrapedPage]


# -----------------------------
# Utility helpers
# -----------------------------

def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _drop_junk_lines(text: str) -> str:
    """
    Heuristic cleanup:
    - Remove very short lines (nav/menu noise)
    - Remove lines with mostly non-letters
    - Keep paragraphs with enough signal
    """
    lines = [ln.strip() for ln in text.splitlines()]
    kept = []
    for ln in lines:
        if not ln:
            continue
        # drop cookie banners / nav-ish snippets (basic heuristics)
        if len(ln) < 30:
            continue
        # if the line is mostly punctuation/symbols/numbers
        alpha = sum(ch.isalpha() for ch in ln)
        if alpha < max(10, int(0.2 * len(ln))):
            continue
        kept.append(ln)
    # Re-paragraph
    return "\n\n".join(kept).strip()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _safe_url(url: str) -> bool:
    if not url:
        return False
    url = url.lower()
    if url.startswith("http://") or url.startswith("https://"):
        return True
    return False


# -----------------------------
# Main tool
# -----------------------------

class WebSearchTool:
    """
    Playwright-based web search + scrape tool.

    Typical usage:
        tool = WebSearchTool(top_k=5, engine="duckduckgo")
        data = tool.search_and_scrape("best places to visit in delhi")
        # data is a dict (json-serializable) with pages + cleaned text
    """

    def __init__(
        self,
        top_k: int = 5,
        engine: str = "duckduckgo",  # "duckduckgo" or "google"
        headless: bool = False,
        nav_timeout_ms: int = 25_000,
        page_timeout_ms: int = 25_000,
        max_chars_per_page: int = 20_000,  # keep responses bounded
        user_agent: Optional[str] = None,
        pause_on_captcha: bool = False,
    ):
        self.top_k = top_k
        self.engine = engine
        self.headless = headless
        self.nav_timeout_ms = nav_timeout_ms
        self.page_timeout_ms = page_timeout_ms
        self.max_chars_per_page = max_chars_per_page
        self.user_agent = user_agent
        self.pause_on_captcha = pause_on_captcha

    def search_and_scrape(self, query: str) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")

        # Try primary engine, fallback to the other
        engines_to_try = [self.engine]
        if self.engine == "duckduckgo":
            engines_to_try.append("google")
        else:
            engines_to_try.append("duckduckgo")

        last_err = None
        for eng in engines_to_try:
            try:
                result = self._run(query=query, engine=eng)
                return asdict(result)
            except Exception as e:
                last_err = e

        raise RuntimeError(f"WebSearchTool failed on all engines. Last error: {last_err}")

    # -----------------------------
    # Internal
    # -----------------------------

    def _run(self, query: str, engine: str) -> WebSearchResult:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent=self.user_agent or (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )
            page = context.new_page()
            page.set_default_navigation_timeout(self.nav_timeout_ms)
            page.set_default_timeout(self.page_timeout_ms)

            urls = self._search_urls(page, query, engine, self.top_k)

            results: List[ScrapedPage] = []
            for i, url in enumerate(urls, start=1):
                scraped = self._scrape_url(context, url, rank=i)
                if scraped:
                    results.append(scraped)
                # be polite-ish
                time.sleep(0.2)

            context.close()
            browser.close()

        return WebSearchResult(query=query, engine=engine, results=results)

    def _search_urls(self, page, query: str, engine: str, top_k: int) -> List[str]:
        if engine == "duckduckgo":
            return self._search_duckduckgo(page, query, top_k)
        elif engine == "google":
            return self._search_google(page, query, top_k)
        else:
            raise ValueError(f"Unknown engine: {engine}")

    def _search_duckduckgo(self, page, query: str, top_k: int) -> List[str]:
        page.goto("https://duckduckgo.com/", wait_until="domcontentloaded")
        page.fill("input[name='q']", query)
        page.keyboard.press("Enter")
        page.wait_for_load_state("domcontentloaded")
        try:
            page.wait_for_selector("a[data-testid='result-title-a'], a.result__a", timeout=10_000)
        except Exception:
            pass

        # DDG results: links typically have data-testid="result-title-a"
        anchors = page.query_selector_all("a[data-testid='result-title-a']")
        urls: List[str] = []
        for a in anchors:
            href = a.get_attribute("href") or ""
            if _safe_url(href) and href not in urls:
                urls.append(href)
            if len(urls) >= top_k:
                break

        # fallback selector if DDG changes markup
        if len(urls) < top_k:
            anchors = page.query_selector_all("a.result__a")
            for a in anchors:
                href = a.get_attribute("href") or ""
                if _safe_url(href) and href not in urls:
                    urls.append(href)
                if len(urls) >= top_k:
                    break

        return urls[:top_k]

    def _search_google(self, page, query: str, top_k: int) -> List[str]:
        page.goto("https://www.google.com/", wait_until="domcontentloaded")

        # Consent dialogs vary; attempt to dismiss if present
        self._try_click_google_consent(page)

        page.fill("textarea[name='q'], input[name='q']", query)
        page.keyboard.press("Enter")
        page.wait_for_load_state("domcontentloaded")
        self._maybe_pause_for_google_captcha(page)
        try:
            page.wait_for_selector("a:has(h3)", timeout=10_000)
        except Exception:
            pass

        anchors = page.query_selector_all("a:has(h3)")
        urls: List[str] = []
        for a in anchors:
            href = a.get_attribute("href") or ""
            if _safe_url(href) and "google.com" not in href and href not in urls:
                urls.append(href)
            if len(urls) >= top_k:
                break

        return urls[:top_k]

    def _maybe_pause_for_google_captcha(self, page) -> None:
        if not self._is_google_captcha(page):
            return

        if not self.pause_on_captcha or self.headless:
            raise RuntimeError("Google captcha detected. Try DuckDuckGo or run headed with pause.")

        print("Google captcha detected. Solve it in the browser, then press Enter to continue...")
        try:
            input()
        except Exception:
            pass

    def _is_google_captcha(self, page) -> bool:
        try:
            url = page.url.lower()
            if "sorry" in url or "captcha" in url or "consent" in url:
                return True
        except Exception:
            pass

        selectors = [
            "iframe[src*='recaptcha']",
            "iframe[title*='reCAPTCHA']",
            "text=I'm not a robot",
            "#recaptcha",
            "#captcha",
        ]
        for sel in selectors:
            try:
                if page.query_selector(sel):
                    return True
            except Exception:
                continue

        return False

    def _try_click_google_consent(self, page) -> None:
        # Best-effort: ignore errors
        try:
            # Buttons can vary by region/language
            possible_selectors = [
                "button:has-text('I agree')",
                "button:has-text('Accept all')",
                "button:has-text('Accept')",
                "button:has-text('Agree')",
                "button:has-text('Accept everything')",
            ]
            for sel in possible_selectors:
                btn = page.query_selector(sel)
                if btn:
                    btn.click()
                    page.wait_for_load_state("domcontentloaded")
                    break
        except Exception:
            pass

    def _scrape_url(self, context, url: str, rank: int) -> Optional[ScrapedPage]:
        page = context.new_page()
        page.set_default_navigation_timeout(self.nav_timeout_ms)
        page.set_default_timeout(self.page_timeout_ms)

        try:
            page.goto(url, wait_until="domcontentloaded")
        except PlaywrightTimeoutError:
            # try a looser wait
            try:
                page.goto(url, wait_until="load")
            except Exception:
                page.close()
                return None
        except Exception:
            page.close()
            return None

        # Try to remove obvious junk elements
        try:
            page.evaluate(
                """() => {
                    const selectors = [
                        'nav', 'header', 'footer', 'aside',
                        '[role="navigation"]',
                        '.cookie', '#cookie', '.cookies', '#cookies',
                        '.subscribe', '.subscription', '.newsletter',
                        '.modal', '.popup', '.advert', '.ads', '[aria-label="advertisement"]'
                    ];
                    for (const sel of selectors) {
                        document.querySelectorAll(sel).forEach(el => el.remove());
                    }
                }"""
            )
        except Exception:
            pass

        # Extract title
        try:
            title = page.title() or url
        except Exception:
            title = url

        # Prefer main/article content when possible; fallback to body innerText
        text = ""
        for selector in ["main", "article", "[role='main']"]:
            try:
                handle = page.query_selector(selector)
                if handle:
                    text = handle.inner_text()
                    if text and len(text) > 500:
                        break
            except Exception:
                continue

        if not text:
            try:
                text = page.inner_text("body")
            except Exception:
                text = ""

        page.close()

        text = _normalize_whitespace(text)
        text = _drop_junk_lines(text)

        if not text:
            return None

        # Bound size to keep downstream LLM summarization sane
        if len(text) > self.max_chars_per_page:
            text = text[: self.max_chars_per_page].rsplit("\n", 1)[0].strip()

        wc = len(text.split())
        return ScrapedPage(
            rank=rank,
            title=title.strip(),
            url=url,
            text=text,
            word_count=wc,
            content_hash=_hash_text(text),
        )


# -----------------------------
# CLI smoke test (optional)
# -----------------------------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str)
    ap.add_argument("--engine", type=str, default="google", choices=["duckduckgo", "google"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--headed", action="store_true", help="Run browser headed (not headless)")
    ap.add_argument("--pause-on-captcha", action="store_true", help="Pause on Google captcha")
    ap.add_argument("--save-json", type=str, default=None, help="Optional path to save output JSON")
    args = ap.parse_args()

    tool = WebSearchTool(
        top_k=args.topk,
        engine=args.engine,
        headless=not args.headed,
        pause_on_captcha=args.pause_on_captcha,
    )
    out = tool.search_and_scrape(args.query)
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
