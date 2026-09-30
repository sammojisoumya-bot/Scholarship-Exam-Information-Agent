"""Run with:  pytest -q

Offline tests only - no API key or network needed. Dates are fixed for stability."""

from datetime import date

import pytest

from rag_agent import Retriever, ScholarshipAgent, chunk_text, parse_records

TODAY = date(2026, 9, 29)


@pytest.fixture(scope="module")
def agent():
    return ScholarshipAgent("college_info.txt", api_key=None)


def test_parse_records_reads_fields():
    recs = parse_records(
        "## Demo\nType: Scholarship\nMin CGPA: 7.5\nDeadline: 2026-10-01"
    )

    assert recs[0].name == "Demo" and recs[0].min_cgpa == 7.5
    assert recs[0].deadline == date(2026, 10, 1)


def test_plain_sections_are_not_records():
    assert parse_records(
        "## Rules\nJust some plain text with no fields."
    ) == []


def test_chunking_skips_comments():
    assert chunk_text(
        "# NOTE: skip\n\nHello world."
    ) == ["Hello world."]


def test_retriever_finds_relevant_chunk():
    r = Retriever([
        "Bananas are yellow.",
        "Revaluation fee is Rs 300 per paper."
    ])

    assert "Revaluation" in r.search(
        "revaluation fee",
        k=1
    )[0][0]


def test_agent_offline_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert ScholarshipAgent(
        "college_info.txt",
        api_key=None
    ).online is False


def test_ask_returns_sources(agent):
    res = agent.ask(
        "What is the revaluation fee?",
        today=TODAY
    )

    assert res["sources"] and "300" in res["answer"]


def test_ask_unknown_topic(agent):
    assert agent.ask(
        "zzzz qqqq xxxx",
        today=TODAY
    )["sources"] == []


def test_high_achiever_is_eligible_for_merit(agent):
    res = agent.check_eligibility(
        9.0,
        500_000,
        "General",
        today=TODAY
    )

    assert "Sunrise Merit Scholarship" in [
        e["name"] for e in res["eligible"]
    ]


def test_income_limit_blocks_need_based(agent):
    res = agent.check_eligibility(
        9.0,
        500_000,
        "General",
        today=TODAY
    )

    blocked = {
        n["name"]: n["reasons"]
        for n in res["not_eligible"]
    }

    assert any(
        "income" in r.lower()
        for r in blocked["Sunrise Need-Based Grant"]
    )


def test_category_restriction(agent):
    general = agent.check_eligibility(
        7.0,
        200_000,
        "General",
        today=TODAY
    )

    obc = agent.check_eligibility(
        7.0,
        200_000,
        "OBC",
        today=TODAY
    )

    name = "Category Fee Support Scheme"

    assert name not in [
        e["name"] for e in general["eligible"]
    ]

    assert name in [
        e["name"] for e in obc["eligible"]
    ]


def test_closed_scholarship_flagged(agent):
    res = agent.check_eligibility(
        9.0,
        100_000,
        "General",
        today=TODAY
    )

    sports = next(
        e for e in res["eligible"]
        if e["name"] == "Sports Excellence Scholarship"
    )

    assert sports["status"] == "Closed"


def test_upcoming_events_sorted_and_windowed(agent):
    events = agent.upcoming_events(
        days=30,
        today=TODAY
    )

    dates = [
        e["date"]
        for e in events
    ]

    assert dates == sorted(dates)

    assert all(
        TODAY <= d <= date(2026, 10, 29)
        for d in dates
    )

    assert any(
        e["name"] == "Scholarship Qualifier Test"
        for e in events
    )