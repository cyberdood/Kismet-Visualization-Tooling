#!/usr/bin/env python3
"""
AI Context Enricher (Autonomous WIDS Agent)

- Poll Elasticsearch for WIDS feature docs missing `context.summary`
- Generate a concise security context + threat analysis + mitigations using Ollama
- Write results back into the same Elasticsearch document

This is designed to run as a small worker container (one process, loop).

Env vars (typical):
  ES_URL=https://es01:9200
  ES_INDEX=wids-wireless-features
  ES_USERNAME=elastic
  ES_PASSWORD=...
  ES_VERIFY_CERTS=false

  OLLAMA_URL=http://ollama:11434
  OLLAMA_MODEL=llama3.1

Optional:
  POLL_SECONDS=10
  BATCH_SIZE=10
  TIME_WINDOW=now-24h
  WRITE_STRUCTURED_CONTEXT=true
"""

import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from elasticsearch import Elasticsearch

# ----------------------- Config -----------------------

ES_URL = os.getenv("ES_URL", "https://es01:9200")
ES_INDEX = os.getenv("ES_INDEX", "wids-wireless-features")

ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_API_KEY = os.getenv("ES_API_KEY")  # optional; overrides basic auth if set
ES_VERIFY_CERTS = os.getenv("ES_VERIFY_CERTS", "false").lower() == "true"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
TIME_WINDOW = os.getenv("TIME_WINDOW", "now-24h")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

WRITE_STRUCTURED_CONTEXT = os.getenv("WRITE_STRUCTURED_CONTEXT", "true").lower() == "true"

# LLM tuning
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

# Prevent repeated rapid failures from hammering endpoints
ERROR_BACKOFF_SECONDS = int(os.getenv("ERROR_BACKOFF_SECONDS", "5"))

# ----------------------- Logging -----------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("context_enricher")


# ----------------------- Elasticsearch -----------------------

def build_es_client() -> Elasticsearch:
    kwargs: Dict[str, Any] = {"verify_certs": ES_VERIFY_CERTS}
    if ES_API_KEY:
        kwargs["api_key"] = ES_API_KEY
    elif ES_USERNAME and ES_PASSWORD:
        kwargs["basic_auth"] = (ES_USERNAME, ES_PASSWORD)

    es = Elasticsearch(ES_URL, **kwargs)
    info = es.info()
    log.info("Connected to Elasticsearch cluster: %s", info.get("cluster_name", "unknown"))
    return es


def find_docs_missing_context(es: Elasticsearch) -> List[Dict[str, Any]]:
    """Find recent docs missing `context.summary`."""
    query = {
        "bool": {
            "filter": [{"range": {"@timestamp": {"gte": TIME_WINDOW}}}],
            "must_not": [{"exists": {"field": "context.summary"}}],
        }
    }

    resp = es.search(
        index=ES_INDEX,
        size=BATCH_SIZE,
        query=query,
        sort=[{"@timestamp": {"order": "desc"}}],
        _source=True,
    )
    return resp.get("hits", {}).get("hits", [])


def write_context(es: Elasticsearch, doc_id: str, summary: str, structured: Optional[Dict[str, Any]] = None) -> None:
    """Write `context.summary` and optional structured context fields."""
    context_obj: Dict[str, Any] = {
        "summary": summary,
        "summary_generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_model": OLLAMA_MODEL,
    }

    if structured and WRITE_STRUCTURED_CONTEXT:
        for k in ("threat_type", "confidence", "indicators", "mitigations"):
            if k in structured and structured[k] is not None:
                context_obj[k] = structured[k]

    es.update(index=ES_INDEX, id=doc_id, doc={"context": context_obj})
    log.info("Updated doc %s with context.summary", doc_id)


# ----------------------- Ollama -----------------------

SYSTEM_PROMPT = (
    "You are a wireless intrusion detection analyst. "
    "Given a structured WIDS feature document about a Wi-Fi access point or wireless device, "
    "produce concise, actionable security context for an analyst dashboard."
)

def build_llm_input(src: Dict[str, Any]) -> str:
    """Reduce the document to the fields that matter most; force JSON output."""
    important = {
        "@timestamp": src.get("@timestamp"),
        "sensor.id": src.get("sensor.id"),
        "sensor.site": src.get("sensor.site"),
        "bssid": src.get("bssid"),
        "ssid": src.get("ssid"),
        "manuf": src.get("manuf"),
        "channel": src.get("channel"),
        "phyname": src.get("phyname"),
        "first_seen": src.get("first_seen"),
        "last_seen": src.get("last_seen"),
        "client_count": src.get("client_count"),
        "ssid_entropy": src.get("ssid_entropy"),
        "rssi_last": src.get("rssi_last"),
        "rssi_min": src.get("rssi_min"),
        "rssi_max": src.get("rssi_max"),
        "rssi_mean": src.get("rssi_mean"),
        "deauth_count_approx": src.get("deauth_count_approx"),
        "probe_req_count_approx": src.get("probe_req_count_approx"),
        "anomaly_score": src.get("anomaly_score"),
        "anomaly_label": src.get("anomaly_label"),
    }

    return (
        "Analyze this wireless feature document and generate an analyst-facing security explanation.\n\n"
        "Return JSON ONLY with this schema:\n"
        "{\n"
        "  \"context_summary\": \"2-4 sentences\",\n"
        "  \"threat_type\": \"rogue_ap|deauth_attack|scanner|benign|unknown\",\n"
        "  \"confidence\": 0-100,\n"
        "  \"indicators\": [\"3-6 short bullets\"],\n"
        "  \"mitigations\": [\"3-6 short bullets\"]\n"
        "}\n\n"
        "Document:\n"
        f"{json.dumps(important, indent=2)}"
    )

def ollama_chat(prompt: str) -> Optional[Dict[str, Any]]:
    """Call Ollama /api/chat and parse JSON response content."""
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"temperature": OLLAMA_TEMPERATURE},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content", "").strip()
        if not content:
            return None
        return json.loads(content)
    except Exception as e:
        log.error("Ollama call or JSON parse failed: %s", e, exc_info=True)
        return None


def compose_summary(llm_json: Dict[str, Any]) -> str:
    """Create a readable string stored in `context.summary`."""
    ctx = (llm_json.get("context_summary") or "").strip()
    threat = (llm_json.get("threat_type") or "unknown").strip()
    conf = llm_json.get("confidence")
    indicators = llm_json.get("indicators") or []
    mitigations = llm_json.get("mitigations") or []

    def bullets(items: Any) -> str:
        if isinstance(items, list):
            lines = [f"- {str(x).strip()}" for x in items if str(x).strip()]
            return "\n".join(lines)
        s = str(items).strip()
        return f"- {s}" if s else ""

    out = []
    if ctx:
        out.append(ctx)
    out.append(f"\nThreat type: {threat} (confidence: {conf})")
    if indicators:
        out.append("\nIndicators:\n" + bullets(indicators))
    if mitigations:
        out.append("\nRecommended mitigations:\n" + bullets(mitigations))

    return "\n".join(out).strip()


# ----------------------- Main loop -----------------------

def main() -> None:
    es = build_es_client()

    while True:
        try:
            hits = find_docs_missing_context(es)
            if not hits:
                time.sleep(POLL_SECONDS)
                continue

            for h in hits:
                doc_id = h.get("_id")
                src = h.get("_source", {})
                if not doc_id or not src:
                    continue

                prompt = build_llm_input(src)
                llm_json = ollama_chat(prompt)
                if not llm_json:
                    time.sleep(ERROR_BACKOFF_SECONDS)
                    continue

                summary = compose_summary(llm_json)
                write_context(es, doc_id, summary, structured=llm_json)

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            log.info("Stopping context enricher.")
            break
        except Exception as e:
            log.error("Worker loop error: %s", e, exc_info=True)
            time.sleep(ERROR_BACKOFF_SECONDS)


if __name__ == "__main__":
    main()
