"""Optional fail-open Gemini sweep before expensive pipeline work."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from applypilot import config
from applypilot.database import get_connection
from applypilot.llm import LLMClient, get_client

log = logging.getLogger(__name__)

_PROMPT_VERSION = "gemini-prefilter-v1"
_SYSTEM_PROMPT = """You are a conservative job-search prefilter.
Reject only a CLEAR mismatch with the supplied candidate targets. Keep anything plausible or uncertain.

Reject clear cases such as:
- the actual work is outside the target technical role families;
- the posting is internship/co-op/student, senior/staff/leadership, or not full-time;
- it clearly cannot employ the candidate in the target geography/work-authorization context;
- it lacks substantial coding/engineering work when the role requires that focus.

Do not reject merely for missing details, imperfect keyword overlap, or preferred qualifications.
Return only a JSON array with one object per input job:
[{"id": 1, "decision": "keep|reject", "confidence": 0.0, "reason": "short reason"}]
"""


def _defaults(search_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    value = search_cfg.get("defaults", {})
    return value if isinstance(value, Mapping) else {}


def sweep_enabled(search_cfg: Mapping[str, Any]) -> bool:
    value = _defaults(search_cfg).get("llm_sweep_enabled", False)
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(value)


def _number(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


def _signature(search_cfg: Mapping[str, Any], profile: Mapping[str, Any], model: str) -> str:
    relevant = {
        "version": _PROMPT_VERSION,
        "model": model,
        "targets": {
            "title_require_any": search_cfg.get("title_require_any", []),
            "locations": search_cfg.get("location_accept", []),
            "defaults": dict(_defaults(search_cfg)),
        },
        "profile": {
            "experience": profile.get("experience", {}),
            "work_authorization": profile.get("work_authorization", {}),
            "target_roles": profile.get("target_roles", {}),
            "technical_focus": profile.get("technical_focus", []),
        },
    }
    raw = json.dumps(relevant, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _candidate_context(search_cfg: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experience": profile.get("experience", {}),
        "work_authorization": profile.get("work_authorization", {}),
        "target_roles": profile.get("target_roles", {}),
        "technical_focus": profile.get("technical_focus", []),
        "accepted_titles": search_cfg.get("title_require_any", []),
        "accepted_locations": search_cfg.get("location_accept", []),
        "employment_type": (profile.get("experience", {}) or {}).get("target_employment_type", ""),
    }


def _job_payload(job: Mapping[str, Any], item_id: int) -> dict[str, Any]:
    description = str(job.get("description") or job.get("full_description") or "")
    return {
        "id": item_id,
        "title": job.get("title"),
        "company": job.get("site"),
        "location": job.get("location"),
        "description_excerpt": description[:1200],
    }


def _parse_response(raw: str) -> dict[int, dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if not text.startswith("["):
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("Gemini sweep returned no JSON array")
        text = text[start : end + 1]

    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Gemini sweep response must be a JSON array")

    decisions: dict[int, dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, Mapping):
            continue
        try:
            item_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        decision = str(item.get("decision") or "").strip().casefold()
        if decision not in {"keep", "reject"}:
            continue
        decisions[item_id] = {
            "decision": decision,
            "confidence": _number(item.get("confidence"), 0.0, minimum=0.0, maximum=1.0),
            "reason": str(item.get("reason") or "").strip()[:500],
        }
    return decisions


def _clear_disabled_sweep(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "SELECT url FROM jobs WHERE filter_reason LIKE 'sweep:%'"
    ).fetchall()
    if rows:
        conn.execute(
            "UPDATE jobs SET filter_reason = NULL, prefilter_decision = NULL, "
            "prefilter_reason = NULL, prefiltered_at = NULL, prefilter_signature = NULL "
            "WHERE filter_reason LIKE 'sweep:%'"
        )
        conn.commit()
    return len(rows)


def run_prefilter(
    conn: sqlite3.Connection | None = None,
    search_cfg: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Sweep unprocessed jobs in Gemini batches; any failure keeps the job."""
    conn = conn or get_connection()
    cfg = search_cfg if search_cfg is not None else config.load_search_config()
    if not sweep_enabled(cfg):
        cleared = _clear_disabled_sweep(conn)
        return {"enabled": False, "scanned": 0, "kept": 0, "rejected": 0, "errors": 0, "cleared": cleared}

    if profile is None:
        try:
            profile = config.load_profile()
        except (FileNotFoundError, OSError, ValueError):
            profile = {}

    defaults = _defaults(cfg)
    model = str(defaults.get("llm_sweep_model") or "gemini/gemini-3.7-flash").strip()
    if not model.startswith("gemini/"):
        raise ValueError("defaults.llm_sweep_model must use the gemini/ provider")
    batch_size = int(_number(defaults.get("llm_sweep_batch_size"), 12, minimum=1, maximum=50))
    reject_confidence = _number(
        defaults.get("llm_sweep_reject_confidence"), 0.90, minimum=0.50, maximum=1.0
    )
    signature = _signature(cfg, profile, model)

    rows = conn.execute(
        "SELECT * FROM jobs WHERE (filter_reason IS NULL OR filter_reason LIKE 'sweep:%') "
        "AND (prefilter_signature IS NULL OR prefilter_signature != ?) "
        "ORDER BY discovered_at DESC",
        (signature,),
    ).fetchall()
    jobs = [dict(row) for row in rows]
    stats: dict[str, Any] = {
        "enabled": True,
        "model": model,
        "scanned": len(jobs),
        "kept": 0,
        "rejected": 0,
        "errors": 0,
        "cleared": 0,
    }
    if not jobs:
        return stats

    llm = client or get_client(model=model)
    candidate = _candidate_context(cfg, profile)
    now = datetime.now(timezone.utc).isoformat()

    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        payload = [_job_payload(job, index + 1) for index, job in enumerate(batch)]
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"candidate": candidate, "jobs": payload}, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ]
        try:
            decisions = _parse_response(
                llm.chat(messages, max_output_tokens=max(512, len(batch) * 100), temperature=0.0)
            )
        except Exception as exc:
            log.warning("Gemini prefilter batch failed open: %s", exc)
            decisions = {}
            stats["errors"] += len(batch)

        for item_id, job in enumerate(batch, 1):
            result = decisions.get(item_id)
            reject = bool(
                result
                and result["decision"] == "reject"
                and result["confidence"] >= reject_confidence
            )
            if reject:
                reason = result["reason"] or "high-confidence semantic mismatch"
                filter_reason = f"sweep:{reason}"
                decision = "reject"
                stats["rejected"] += 1
            else:
                reason = result["reason"] if result else "fail-open: missing or invalid decision"
                filter_reason = None
                decision = "keep"
                stats["kept"] += 1

            conn.execute(
                "UPDATE jobs SET filter_reason = ?, prefilter_decision = ?, prefilter_reason = ?, "
                "prefiltered_at = ?, prefilter_signature = ? WHERE url = ?",
                (filter_reason, decision, reason, now, signature, job["url"]),
            )
        conn.commit()

    log.info(
        "Gemini sweep: %d scanned, %d kept, %d rejected, %d fail-open errors",
        stats["scanned"], stats["kept"], stats["rejected"], stats["errors"],
    )
    return stats
