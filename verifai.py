"""
VerifAI — AI Research Verification Pipeline
============================================
Input  : A markdown table with columns: Claim | Source URL | Source Name
Output : A detailed verification report (CSV + printed summary)

Usage:
    python verifai.py --input research.md
    python verifai.py --input research.md --output report.csv
    python verifai.py --demo   (runs on built-in sample data)

Get a free Groq API key at: https://console.groq.com
"""

from openai import OpenAI
import httpx
import trafilatura
import requests
import csv
import json
import argparse
import sys
import time
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime


GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"

# Content thresholds
THRESHOLD_INSUFFICIENT = 300   # below this: don't verify, mark as insufficient
THRESHOLD_LOW_CONTENT  = 800   # between 300-800: verify but cap confidence at 60%


# --- Step 1: Parse the input markdown table ----------------------------------

def parse_markdown_table(text: str) -> list:
    claims = []
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in lines:
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[-| ]+\|$", line):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 2:
            continue
        claim_text = parts[0].strip()
        source_url = parts[1].strip() if len(parts) > 1 else ""
        source_name = parts[2].strip() if len(parts) > 2 else source_url
        if not claim_text or claim_text.lower() in ("claim", "claims"):
            continue
        claims.append({
            "claim": claim_text,
            "source_url": source_url,
            "source_name": source_name,
        })
    return claims


# --- Step 2: Fetch and extract page content ----------------------------------

def fetch_page_content(url: str, timeout: int = 15):
    if not url.startswith("http"):
        url = "https://" + url
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        if response.status_code in (401, 407, 429):
            return "", "inaccessible"
        if response.status_code >= 400 and response.status_code != 403:
            return "", "inaccessible"
        content = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if not content or len(content.strip()) < 100:
            content = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=True,
                no_fallback=True,
                favor_recall=True,
            )
        if not content or len(content.strip()) < 50:
            return "", "empty"
        return content[:8000], "ok"
    except requests.exceptions.Timeout:
        return "", "inaccessible"
    except requests.exceptions.ConnectionError:
        return "", "inaccessible"
    except Exception as e:
        print(f"        DEBUG ERROR: {e}")
        return "", "inaccessible"


# --- Step 3: Ask Groq to verify the claim ------------------------------------

VERIFICATION_PROMPT = """You are a research fact-checker. Your job is to verify whether a specific claim is supported by the text from a source webpage.

CLAIM:
{claim}

SOURCE NAME: {source_name}
SOURCE URL: {source_url}

EXTRACTED PAGE TEXT ({char_count} characters extracted):
{page_content}

{content_warning}
---

Carefully read the page text and assess whether it supports, contradicts, or is unrelated to the claim.

You MUST return ONLY a valid JSON object. No explanation before or after. No markdown. Start your response with {{ and end with }}.

The JSON must have exactly these fields:

{{
  "verdict": "<one of: confirmed | partial | inferred | not_found>",
  "snippet": "<choose based on verdict: if confirmed or partial — copy the exact sentence from the page that most directly proves the claim; if inferred — copy the sentence that most closely relates to the claim even though it does not directly state it; if not_found — copy a sentence that shows the claim is absent or contradicted. Max 200 chars. If truly nothing relevant exists, write: No relevant text found on page.>",
  "reasoning": "<1-2 sentence explanation of your verdict. Be specific about what matched, what did not, or what was overgeneralised>",
  "confidence": <integer 0-100>,
  "confidence_label": "<one of: High | Medium | Low>"
}}

Verdict definitions:
- confirmed   : The claim is directly and accurately stated in the source
- partial     : The claim is related to content in the source but is oversimplified, missing context, or slightly inaccurate
- inferred    : The source does not state the claim directly but it could be reasonably interpreted from the content
- not_found   : The claim is not supported by this source, or the source contradicts it

Confidence guidelines:
- High (75-100)  : Strong textual match
- Medium (40-74) : Partial match or some ambiguity
- Low (0-39)     : Weak or no match

IMPORTANT: Return ONLY the JSON object. Nothing else."""

CONTENT_WARNING_LOW = """WARNING: Only {char_count} characters were extracted from this page — this is a low amount of content. 
The page may be a dashboard, redirect, or JavaScript-heavy site that could not be fully scraped.
You MUST reflect this uncertainty in your confidence score — cap your confidence at 60% maximum regardless of how well the text seems to match."""

CONTENT_WARNING_NONE = ""


def extract_json_from_response(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*?"verdict".*?\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end+1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response: {raw[:200]}")


def verify_claim_with_groq(claim, source_url, source_name, page_content, client, retries=2):
    char_count = len(page_content)

    # Build content warning based on how much text was extracted
    if char_count < THRESHOLD_LOW_CONTENT:
        content_warning = CONTENT_WARNING_LOW.format(char_count=char_count)
    else:
        content_warning = CONTENT_WARNING_NONE

    prompt = VERIFICATION_PROMPT.format(
        claim=claim,
        source_url=source_url,
        source_name=source_name,
        page_content=page_content,
        char_count=char_count,
        content_warning=content_warning,
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            result = extract_json_from_response(raw)

            required = {"verdict", "snippet", "reasoning", "confidence", "confidence_label"}
            if not required.issubset(result.keys()):
                raise ValueError(f"Missing fields: {required - result.keys()}")

            if result["verdict"] not in {"confirmed", "partial", "inferred", "not_found"}:
                result["verdict"] = "not_found"

            result["confidence"] = max(0, min(100, int(result["confidence"])))

            # Enforce confidence cap for low content pages
            if char_count < THRESHOLD_LOW_CONTENT:
                if result["confidence"] > 60:
                    result["confidence"] = 60
                result["confidence_label"] = (
                    "Medium" if result["confidence"] >= 40 else "Low"
                )

            return result

        except Exception as e:
            if attempt < retries - 1:
                print(f"        Retry {attempt + 1}/{retries - 1} — retrying...")
                time.sleep(1)
            else:
                return {
                    "verdict": "not_found",
                    "snippet": "Could not parse model response after retries.",
                    "reasoning": f"Verification failed: {str(e)[:100]}",
                    "confidence": 0,
                    "confidence_label": "Low",
                }


# --- Step 4: Run the full pipeline -------------------------------------------

def run_pipeline(claims: list, api_key: str) -> list:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        http_client=httpx.Client(verify=False),
    )

    results = []
    total = len(claims)
    print(f"\nVerifAI — verifying {total} claim(s)\n{'─' * 50}")

    for i, item in enumerate(claims, 1):
        claim = item["claim"]
        url = item["source_url"]
        name = item["source_name"]

        print(f"[{i}/{total}] {claim[:70]}...")
        print(f"        Fetching: {url}")

        page_content, fetch_status = fetch_page_content(url)

        if fetch_status == "inaccessible":
            print(f"        Status: Source inaccessible (paywalled or blocked)\n")
            results.append({
                **item,
                "fetch_status": "inaccessible",
                "verdict": "inaccessible",
                "snippet": "Source could not be accessed — may be paywalled, rate-limited, or blocking scrapers.",
                "reasoning": "Page was inaccessible. Verdict cannot be determined.",
                "confidence": 0,
                "confidence_label": "Low",
                "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            continue

        if fetch_status == "empty":
            print(f"        Status: Page fetched but no readable content extracted\n")
            results.append({
                **item,
                "fetch_status": "empty",
                "verdict": "not_found",
                "snippet": "Page was fetched but no readable text could be extracted.",
                "reasoning": "Page content could not be extracted — may be a dynamic or media-heavy page.",
                "confidence": 0,
                "confidence_label": "Low",
                "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            continue

        char_count = len(page_content)

        # Below minimum threshold — don't verify, flag as insufficient
        if char_count < THRESHOLD_INSUFFICIENT:
            print(f"        Status: Insufficient content ({char_count} chars) — too little to verify reliably\n")
            results.append({
                **item,
                "fetch_status": "insufficient",
                "verdict": "insufficient_source",
                "snippet": f"Only {char_count} characters extracted — page is likely a dashboard, redirect, or JavaScript-heavy site.",
                "reasoning": "Not enough content was extracted from this page to verify the claim reliably. Try a different source URL.",
                "confidence": 0,
                "confidence_label": "Low",
                "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            continue

        # Between 300-800 chars — verify but warn
        if char_count < THRESHOLD_LOW_CONTENT:
            print(f"        Status: Low content ({char_count} chars) — verifying with caution...")
        else:
            print(f"        Status: Page fetched ({char_count} chars). Verifying with Groq...")

        verdict_data = verify_claim_with_groq(claim, url, name, page_content, client)
        print(f"        Verdict: {verdict_data['verdict'].upper()} | Confidence: {verdict_data['confidence']}% ({verdict_data['confidence_label']})\n")

        results.append({
            **item,
            "fetch_status": "ok" if char_count >= THRESHOLD_LOW_CONTENT else "low_content",
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            **verdict_data,
        })

        if i < total:
            time.sleep(0.3)

    return results


# --- Step 5: Output ----------------------------------------------------------

def print_summary(results: list):
    counts = {
        "confirmed": 0, "partial": 0, "inferred": 0,
        "not_found": 0, "inaccessible": 0, "insufficient_source": 0
    }
    for r in results:
        v = r.get("verdict", "not_found")
        counts[v] = counts.get(v, 0) + 1

    total = len(results)
    print(f"\n{'=' * 50}")
    print(f"  VERIFAI RESULTS SUMMARY — {total} claims")
    print(f"{'=' * 50}")
    print(f"  Confirmed          : {counts['confirmed']}")
    print(f"  Partially correct  : {counts['partial']}")
    print(f"  Inferred           : {counts['inferred']}")
    print(f"  Not found          : {counts['not_found']}")
    print(f"  Inaccessible       : {counts['inaccessible']}")
    print(f"  Insufficient source: {counts['insufficient_source']}")
    print(f"{'─' * 50}")

    for r in results:
        print(f"\n  Claim   : {r['claim'][:80]}")
        print(f"  Source  : {r['source_name']}")
        print(f"  Verdict : {r['verdict'].upper()} | Confidence: {r.get('confidence', 0)}% ({r.get('confidence_label', '')})")
        print(f"  Snippet : {r.get('snippet', '')[:200]}")
        print(f"  Reason  : {r.get('reasoning', '')[:200]}")

    print(f"\n{'=' * 50}\n")


def save_csv(results: list, output_path: str):
    fields = [
        "claim", "source_name", "source_url", "checked_at",
        "verdict", "confidence", "confidence_label",
        "snippet", "reasoning", "fetch_status",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  Report saved to: {output_path}")


# --- Demo data ----------------------------------------------------------------

DEMO_INPUT = """
| Claim | Source URL | Source Name |
|-------|-----------|-------------|
| The Eiffel Tower is located in Paris | https://simple.wikipedia.org/wiki/Eiffel_Tower | Simple Wikipedia |
| Python was created by Guido van Rossum | https://simple.wikipedia.org/wiki/Python_(programming_language) | Simple Wikipedia |
| The FIFA World Cup is held every four years | https://simple.wikipedia.org/wiki/FIFA_World_Cup | Simple Wikipedia |
"""


# --- Entry point -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VerifAI — AI research claim verifier")
    parser.add_argument("--input", help="Path to markdown table file (.md or .txt)")
    parser.add_argument("--output", help="Path to save CSV report (optional)", default=None)
    parser.add_argument("--demo", action="store_true", help="Run on built-in demo data")
    parser.add_argument("--api-key", help="Groq API key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or GROQ_API_KEY
    if api_key == "YOUR_GROQ_API_KEY_HERE":
        print("\nError: No API key found.")
        print("Get a free key at: https://console.groq.com")
        print("Then either:")
        print("  1. Set GROQ_API_KEY at the top of this script, or")
        print("  2. Pass it as: python verifai.py --demo --api-key YOUR_KEY\n")
        sys.exit(1)

    if args.demo:
        print("Running in demo mode with sample claims...")
        raw_text = DEMO_INPUT
    elif args.input:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args.input}")
            sys.exit(1)
    else:
        print("No input provided. Use --input <file> or --demo")
        print("Run: python verifai.py --help")
        sys.exit(1)

    claims = parse_markdown_table(raw_text)
    if not claims:
        print("Error: No claims found. Check your markdown table format.")
        print("Expected:\n| Claim | Source URL | Source Name |")
        sys.exit(1)

    print(f"Parsed {len(claims)} claim(s) from input.")
    results = run_pipeline(claims, api_key)
    print_summary(results)

    output_path = args.output or f"verifai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_csv(results, output_path)


if __name__ == "__main__":
    main()