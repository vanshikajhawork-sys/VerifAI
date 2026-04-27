"""
VerifAI — Flask API Server
==========================
Wraps the verification pipeline and exposes it as a REST API.

Usage:
    python server.py

Then open index.html in your browser.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
import httpx
import trafilatura
import requests as req
import json
import time
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime

app = Flask(__name__)
CORS(app)

import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

THRESHOLD_INSUFFICIENT = 300
THRESHOLD_LOW_CONTENT  = 800


# --- Markdown table parser ---------------------------------------------------

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


# --- Page fetcher ------------------------------------------------------------

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
        response = req.get(url, headers=headers, timeout=timeout, verify=False)
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
    except req.exceptions.Timeout:
        return "", "inaccessible"
    except req.exceptions.ConnectionError:
        return "", "inaccessible"
    except Exception:
        return "", "inaccessible"


# --- Groq verifier -----------------------------------------------------------

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
    raise ValueError(f"Could not extract JSON: {raw[:200]}")


def verify_claim(claim, source_url, source_name, page_content, client):
    char_count = len(page_content)
    content_warning = (
        CONTENT_WARNING_LOW.format(char_count=char_count)
        if char_count < THRESHOLD_LOW_CONTENT else ""
    )
    prompt = VERIFICATION_PROMPT.format(
        claim=claim,
        source_url=source_url,
        source_name=source_name,
        page_content=page_content,
        char_count=char_count,
        content_warning=content_warning,
    )
    for attempt in range(2):
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
                raise ValueError("Missing fields")
            if result["verdict"] not in {"confirmed", "partial", "inferred", "not_found"}:
                result["verdict"] = "not_found"
            result["confidence"] = max(0, min(100, int(result["confidence"])))
            if char_count < THRESHOLD_LOW_CONTENT:
                if result["confidence"] > 60:
                    result["confidence"] = 60
                result["confidence_label"] = "Medium" if result["confidence"] >= 40 else "Low"
            return result
        except Exception as e:
            if attempt == 1:
                return {
                    "verdict": "not_found",
                    "snippet": "Could not parse model response.",
                    "reasoning": f"Verification failed: {str(e)[:100]}",
                    "confidence": 0,
                    "confidence_label": "Low",
                }
            time.sleep(1)


# --- API endpoint ------------------------------------------------------------

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/verify", methods=["POST"])
def verify():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    raw_text = data["text"]
    claims = parse_markdown_table(raw_text)

    if not claims:
        return jsonify({"error": "No claims found. Check your markdown table format."}), 400

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        http_client=httpx.Client(verify=False),
    )

    results = []

    for item in claims:
        claim = item["claim"]
        url = item["source_url"]
        name = item["source_name"]

        page_content, fetch_status = fetch_page_content(url)

        if fetch_status == "inaccessible":
            results.append({
                **item,
                "verdict": "inaccessible",
                "snippet": "Source could not be accessed — may be paywalled or blocking scrapers.",
                "reasoning": "Page was inaccessible. Verdict cannot be determined.",
                "confidence": 0,
                "confidence_label": "Low",
                "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "fetch_status": "inaccessible",
            })
            continue

        if fetch_status == "empty":
            results.append({
                **item,
                "verdict": "not_found",
                "snippet": "Page was fetched but no readable text could be extracted.",
                "reasoning": "Page content could not be extracted — may be a dynamic or media-heavy page.",
                "confidence": 0,
                "confidence_label": "Low",
                "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "fetch_status": "empty",
            })
            continue

        char_count = len(page_content)

        if char_count < THRESHOLD_INSUFFICIENT:
            results.append({
                **item,
                "verdict": "insufficient_source",
                "snippet": f"Only {char_count} characters extracted — page is likely a dashboard or JavaScript-heavy site.",
                "reasoning": "Not enough content extracted to verify reliably. Try a different source URL.",
                "confidence": 0,
                "confidence_label": "Low",
                "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "fetch_status": "insufficient",
            })
            continue

        verdict_data = verify_claim(claim, url, name, page_content, client)

        results.append({
            **item,
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "fetch_status": "ok" if char_count >= THRESHOLD_LOW_CONTENT else "low_content",
            **verdict_data,
        })

        time.sleep(0.3)

    return jsonify({
        "total": len(results),
        "results": results,
        "summary": {
            "confirmed": sum(1 for r in results if r["verdict"] == "confirmed"),
            "partial": sum(1 for r in results if r["verdict"] == "partial"),
            "inferred": sum(1 for r in results if r["verdict"] == "inferred"),
            "not_found": sum(1 for r in results if r["verdict"] == "not_found"),
            "inaccessible": sum(1 for r in results if r["verdict"] == "inaccessible"),
            "insufficient_source": sum(1 for r in results if r["verdict"] == "insufficient_source"),
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "VerifAI server is running"})


if __name__ == "__main__":
    print("\nVerifAI server starting...")
    print("Open index.html in your browser to use the tool\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)