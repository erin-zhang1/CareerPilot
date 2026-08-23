"""Deterministic job filters shared by every pipeline stage."""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from applypilot import config

log = logging.getLogger(__name__)

_JUNIOR_TITLE_RE = re.compile(
    r"\b(?:junior|jr\.?|entry[\s-]*level|new[\s-]*grad(?:uate)?|graduate|"
    r"intern(?:ship)?|co[\s-]*op|level[\s-]*(?:1|i))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FilterSettings:
    """Normalized hard-filter settings from searches.yaml."""

    exclude_titles: tuple[str, ...] = ()
    title_require_any: tuple[str, ...] = ()
    location_accept: tuple[str, ...] = ()
    location_reject_non_remote: tuple[str, ...] = ()
    blocked_countries: tuple[str, ...] = ()
    seniority_floor_years: float = 0
    max_required_years: float = 0


def _string_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def load_filter_settings(search_cfg: Mapping[str, Any] | None = None) -> FilterSettings:
    """Load filters while supporting both current and legacy location layouts."""
    cfg = search_cfg if search_cfg is not None else config.load_search_config()
    defaults = cfg.get("defaults", {}) if isinstance(cfg.get("defaults"), Mapping) else {}
    nested_location = cfg.get("location", {}) if isinstance(cfg.get("location"), Mapping) else {}

    if "location_accept" in cfg:
        location_accept = _string_list(cfg.get("location_accept"))
    else:
        location_accept = _string_list(nested_location.get("accept_patterns"))

    if "location_reject_non_remote" in cfg:
        location_reject = _string_list(cfg.get("location_reject_non_remote"))
    else:
        location_reject = _string_list(nested_location.get("reject_patterns"))

    floor_raw = defaults.get("seniority_floor_years", cfg.get("seniority_floor_years", 0))
    max_required_raw = defaults.get("max_required_years", cfg.get("max_required_years", 0))
    try:
        floor = max(0.0, float(floor_raw or 0))
    except (TypeError, ValueError):
        log.warning("Invalid seniority_floor_years=%r; disabling seniority filter", floor_raw)
        floor = 0.0
    try:
        max_required = max(0.0, float(max_required_raw or 0))
    except (TypeError, ValueError):
        log.warning("Invalid max_required_years=%r; disabling experience filter", max_required_raw)
        max_required = 0.0

    return FilterSettings(
        exclude_titles=_string_list(cfg.get("exclude_titles")),
        title_require_any=_string_list(cfg.get("title_require_any")),
        location_accept=location_accept,
        location_reject_non_remote=location_reject,
        blocked_countries=_string_list(defaults.get("blocked_countries", cfg.get("blocked_countries"))),
        seniority_floor_years=floor,
        max_required_years=max_required,
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = text.casefold()
    normalized_phrase = phrase.casefold().strip()
    if not normalized_phrase:
        return False
    if len(normalized_phrase) <= 3 and normalized_phrase.isalnum():
        return re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text) is not None
    return normalized_phrase in normalized_text


def _contains_country(location: str, country: str) -> bool:
    """Match a country name/code without treating it as part of another word."""
    normalized_country = country.casefold().strip()
    if not normalized_country:
        return False
    pattern = re.escape(normalized_country).replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){pattern}(?!\w)", location.casefold()) is not None


def _candidate_years(profile: Mapping[str, Any] | None) -> float | None:
    if not profile:
        return None
    experience = profile.get("experience", {})
    if not isinstance(experience, Mapping):
        return None
    raw = experience.get("years_of_experience_total")
    match = re.search(r"\d+(?:\.\d+)?", str(raw or ""))
    return float(match.group()) if match else None


_HARD_EXPERIENCE_PATTERNS = (
    re.compile(r"(?<![\d-])(\d{1,2})\s*\+\s*(?:years?|yrs?)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:at\s+least|minimum(?:\s+of)?|(?:must\s+have|requires?)(?:\s+at\s+least)?)\s+"
        r"(\d{1,2})\s*(?:years?|yrs?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(\d{1,2})\s+or\s+more\s+(?:years?|yrs?)\b", re.IGNORECASE),
    re.compile(
        r"\b(\d{1,2})\s*(?:years?|yrs?)(?:\s+of\s+[\w /&+-]{1,40})?\s+"
        r"(?:is\s+)?(?:required|minimum)\b",
        re.IGNORECASE,
    ),
)


def _hard_experience_requirement(text: str) -> float | None:
    """Extract the largest explicitly mandatory years-of-experience value."""
    matches = [
        float(match.group(1))
        for pattern in _HARD_EXPERIENCE_PATTERNS
        for match in pattern.finditer(text)
    ]
    return max(matches) if matches else None


def location_filter_reason(
    location: str | None,
    accept: Sequence[str],
    reject_non_remote: Sequence[str],
) -> str | None:
    """Return a location rejection reason, or None when the location passes."""
    if not location:
        return None
    if any(_contains_phrase(location, marker) for marker in ("remote", "anywhere", "work from home", "wfh", "distributed")):
        return None
    for phrase in reject_non_remote:
        if _contains_phrase(location, phrase):
            return f"location_rejected:{phrase}"
    if accept and not any(_contains_phrase(location, phrase) for phrase in accept):
        return "location_not_accepted"
    return None


def filter_job(
    job: Mapping[str, Any],
    search_cfg: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
) -> str | None:
    """Apply title, seniority, country, and location rules to one job."""
    settings = load_filter_settings(search_cfg)
    title = str(job.get("title") or "").strip()
    location = str(job.get("location") or "").strip()
    description = str(job.get("full_description") or job.get("description") or "")

    if title:
        for phrase in settings.exclude_titles:
            if _contains_phrase(title, phrase):
                return f"rule:title_excluded:{phrase}"

        if settings.title_require_any and not any(
            _contains_phrase(title, phrase) for phrase in settings.title_require_any
        ):
            return "rule:title_not_required"

        years = _candidate_years(profile)
        if (
            settings.seniority_floor_years > 0
            and years is not None
            and years >= settings.seniority_floor_years
            and _JUNIOR_TITLE_RE.search(title)
        ):
            return "rule:seniority_below_target"

    if settings.max_required_years > 0 and description:
        required_years = _hard_experience_requirement(description)
        if required_years is not None and required_years > settings.max_required_years:
            return f"rule:experience_minimum_exceeds:{required_years:g}"

    if location:
        for country in settings.blocked_countries:
            if _contains_country(location, country):
                return f"rule:country_blocked:{country}"

    geo_reason = location_filter_reason(
        location or None,
        settings.location_accept,
        settings.location_reject_non_remote,
    )
    return f"rule:{geo_reason}" if geo_reason else None


def apply_rule_gate(
    conn: sqlite3.Connection,
    search_cfg: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Re-evaluate all unfiltered or rule-filtered database rows."""
    cfg = search_cfg if search_cfg is not None else config.load_search_config()
    if profile is None:
        try:
            profile = config.load_profile()
        except (FileNotFoundError, OSError, ValueError):
            profile = {}

    rows = conn.execute(
        "SELECT url, title, location, description, full_description, filter_reason FROM jobs "
        "WHERE filter_reason IS NULL OR filter_reason LIKE 'rule:%'"
    ).fetchall()
    stats = {"scanned": len(rows), "filtered": 0, "cleared": 0}

    for row in rows:
        job = dict(row) if hasattr(row, "keys") else {
            "url": row[0],
            "title": row[1],
            "location": row[2],
            "description": row[3],
            "full_description": row[4],
            "filter_reason": row[5],
        }
        previous = job.get("filter_reason")
        reason = filter_job(job, cfg, profile)
        if reason:
            stats["filtered"] += 1
        elif previous:
            stats["cleared"] += 1
        if reason != previous:
            conn.execute("UPDATE jobs SET filter_reason = ? WHERE url = ?", (reason, job["url"]))

    conn.commit()
    log.info(
        "Hard rule gate: %d scanned, %d filtered, %d cleared",
        stats["scanned"], stats["filtered"], stats["cleared"],
    )
    return stats
