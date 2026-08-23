from __future__ import annotations

from applypilot.database import close_connection, get_jobs_by_stage, init_db
from applypilot.filters import apply_rule_gate, filter_job, load_filter_settings, location_filter_reason


def _profile(years: str = "5") -> dict:
    return {"experience": {"years_of_experience_total": years}}


def test_load_filter_settings_supports_legacy_nested_location_config() -> None:
    settings = load_filter_settings(
        {
            "location": {
                "accept_patterns": ["Toronto"],
                "reject_patterns": ["onsite only"],
            }
        }
    )

    assert settings.location_accept == ("Toronto",)
    assert settings.location_reject_non_remote == ("onsite only",)


def test_title_exclusion_precedes_allow_list() -> None:
    cfg = {
        "exclude_titles": ["director"],
        "title_require_any": ["engineering"],
    }

    reason = filter_job({"title": "Director of Engineering"}, cfg, _profile())

    assert reason == "rule:title_excluded:director"


def test_title_allow_list_rejects_mismatch_but_allows_unknown_title() -> None:
    cfg = {"title_require_any": ["backend", "platform"]}

    assert filter_job({"title": "Product Designer"}, cfg, _profile()) == "rule:title_not_required"
    assert filter_job({"title": None}, cfg, _profile()) is None


def test_seniority_filter_only_activates_when_candidate_meets_floor() -> None:
    cfg = {"defaults": {"seniority_floor_years": 4}}
    job = {"title": "Junior Backend Engineer"}

    assert filter_job(job, cfg, _profile("3")) is None
    assert filter_job(job, cfg, _profile("5+ years")) == "rule:seniority_below_target"


def test_hard_experience_requirement_rejects_only_values_above_limit() -> None:
    cfg = {"defaults": {"max_required_years": 3}}

    assert filter_job({"description": "Requires 3 years of Python experience."}, cfg) is None
    assert (
        filter_job({"full_description": "Candidates must have at least 4 years experience."}, cfg)
        == "rule:experience_minimum_exceeds:4"
    )
    assert (
        filter_job({"description": "5+ years of backend engineering experience."}, cfg)
        == "rule:experience_minimum_exceeds:5"
    )
    assert filter_job({"description": "0-4 years of experience; new grads welcome."}, cfg) is None
    assert filter_job({"description": "Four years preferred, not required."}, cfg) is None


def test_country_block_is_hard_even_for_remote_and_short_codes_use_boundaries() -> None:
    cfg = {"defaults": {"blocked_countries": ["India", "US"]}}

    assert filter_job({"location": "Remote - India"}, cfg, _profile()) == "rule:country_blocked:India"
    assert filter_job({"location": "Remote, US"}, cfg, _profile()) == "rule:country_blocked:US"
    assert filter_job({"location": "Australia"}, cfg, _profile()) is None
    assert filter_job({"location": "Indiana, USA"}, cfg, _profile()) is None


def test_location_gate_handles_remote_reject_allow_and_empty_config() -> None:
    assert location_filter_reason("Remote - anywhere", ["Toronto"], ["New York"]) is None
    assert location_filter_reason("New York, NY", ["Toronto"], ["New York"]) == "location_rejected:New York"
    assert location_filter_reason("Vancouver, BC", ["Toronto"], []) == "location_not_accepted"
    assert location_filter_reason("Vancouver, BC", [], []) is None


def test_rule_gate_soft_marks_rows_and_pending_queries_skip_them(tmp_path) -> None:
    db_path = tmp_path / "applypilot.db"
    conn = init_db(db_path)
    try:
        rows = [
            ("https://example.com/ok", "Backend Engineer", "Toronto, ON"),
            ("https://example.com/title", "Sales Director", "Toronto, ON"),
            ("https://example.com/country", "Backend Engineer", "Remote - India"),
            ("https://example.com/junior", "Junior Platform Engineer", "Toronto, ON"),
        ]
        conn.executemany("INSERT INTO jobs (url, title, location) VALUES (?, ?, ?)", rows)
        conn.commit()

        stats = apply_rule_gate(
            conn,
            {
                "exclude_titles": ["sales"],
                "title_require_any": ["backend", "platform"],
                "location_accept": ["Toronto"],
                "defaults": {
                    "blocked_countries": ["India"],
                    "seniority_floor_years": 4,
                },
            },
            _profile("6"),
        )

        reasons = dict(conn.execute("SELECT url, filter_reason FROM jobs"))
        pending = get_jobs_by_stage(conn=conn, stage="pending_detail", limit=20)

        assert stats == {"scanned": 4, "filtered": 3, "cleared": 0}
        assert reasons["https://example.com/ok"] is None
        assert reasons["https://example.com/title"] == "rule:title_excluded:sales"
        assert reasons["https://example.com/country"] == "rule:country_blocked:India"
        assert reasons["https://example.com/junior"] == "rule:seniority_below_target"
        assert [job["url"] for job in pending] == ["https://example.com/ok"]
    finally:
        close_connection(db_path)
