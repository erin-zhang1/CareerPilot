from __future__ import annotations

from applypilot.database import close_connection, get_jobs_by_stage, init_db
from applypilot.scoring.prefilter import run_prefilter


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def chat(self, messages, **kwargs) -> str:
        self.calls += 1
        return self.response


def _config(enabled: bool = True) -> dict:
    return {
        "title_require_any": ["software engineer", "data scientist"],
        "location_accept": ["Toronto"],
        "defaults": {
            "llm_sweep_enabled": enabled,
            "llm_sweep_model": "gemini/gemini-3.7-flash",
            "llm_sweep_batch_size": 10,
            "llm_sweep_reject_confidence": 0.9,
        },
    }


def test_prefilter_rejects_only_high_confidence_and_skips_downstream(tmp_path) -> None:
    db_path = tmp_path / "applypilot.db"
    conn = init_db(db_path)
    try:
        conn.executemany(
            "INSERT INTO jobs (url, title, location, description) VALUES (?, ?, ?, ?)",
            [
                ("https://example.com/1", "Software Engineer", "Toronto", "Backend coding role"),
                ("https://example.com/2", "Data Scientist", "Toronto", "Marketing reporting role"),
                ("https://example.com/3", "Data Scientist", "Toronto", "Unclear analytics role"),
            ],
        )
        conn.commit()
        client = FakeClient(
            '[{"id":1,"decision":"keep","confidence":0.99,"reason":"coding"},'
            '{"id":2,"decision":"reject","confidence":0.96,"reason":"not engineering"},'
            '{"id":3,"decision":"reject","confidence":0.70,"reason":"uncertain"}]'
        )

        stats = run_prefilter(conn, _config(), {"experience": {}}, client)
        reasons = dict(conn.execute("SELECT url, filter_reason FROM jobs"))
        pending = get_jobs_by_stage(conn=conn, stage="pending_detail", limit=10)

        assert stats["rejected"] == 1
        assert stats["kept"] == 2
        assert reasons["https://example.com/2"] == "sweep:not engineering"
        assert reasons["https://example.com/1"] is None
        assert reasons["https://example.com/3"] is None
        assert {job["url"] for job in pending} == {
            "https://example.com/1",
            "https://example.com/3",
        }
    finally:
        close_connection(db_path)


def test_prefilter_fails_open_on_malformed_response(tmp_path) -> None:
    db_path = tmp_path / "applypilot.db"
    conn = init_db(db_path)
    try:
        conn.execute(
            "INSERT INTO jobs (url, title, location) VALUES (?, ?, ?)",
            ("https://example.com/1", "Software Engineer", "Toronto"),
        )
        conn.commit()

        stats = run_prefilter(conn, _config(), {}, FakeClient("not json"))
        row = conn.execute(
            "SELECT filter_reason, prefilter_decision FROM jobs WHERE url = ?",
            ("https://example.com/1",),
        ).fetchone()

        assert stats["errors"] == 1
        assert row[0] is None
        assert row[1] == "keep"
    finally:
        close_connection(db_path)


def test_disabling_prefilter_clears_previous_sweep_rejection(tmp_path) -> None:
    db_path = tmp_path / "applypilot.db"
    conn = init_db(db_path)
    try:
        conn.execute(
            "INSERT INTO jobs (url, title, filter_reason, prefilter_decision, prefilter_signature) "
            "VALUES (?, ?, ?, ?, ?)",
            ("https://example.com/1", "Data Scientist", "sweep:not coding", "reject", "old"),
        )
        conn.commit()

        stats = run_prefilter(conn, _config(enabled=False), {})
        row = conn.execute(
            "SELECT filter_reason, prefilter_decision, prefilter_signature FROM jobs"
        ).fetchone()

        assert stats["cleared"] == 1
        assert tuple(row) == (None, None, None)
    finally:
        close_connection(db_path)
