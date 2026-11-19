from __future__ import annotations

import datetime as dt

from datestamp.parse import parse_date
from datestamp.confidence import PlausibilityContext, assess_plausibility


def test_plausibility_accepts_valid_date_in_range() -> None:
    ctx = PlausibilityContext(
        scan_date="2001-06-15",
        today=dt.date(2005, 1, 1),
        archive_year_min=1980,
        archive_year_max=1999,
    )
    res = assess_plausibility("1990-05-09", context=ctx)
    assert res.ok is True
    assert res.parsed_date == dt.date(1990, 5, 9)
    assert res.score > 0.9
    assert any("valid calendar date" in r.lower() for r in res.reasons)
    assert any("plausibility checks passed" in r.lower() for r in res.reasons)


def test_plausibility_rejects_missing_or_invalid_date() -> None:
    res = assess_plausibility("", context=PlausibilityContext(today=dt.date(2005, 1, 1)))
    assert res.ok is False
    assert res.parsed_date is None
    assert res.score == 0.0
    assert any("missing or invalid" in p.lower() for p in res.penalties)

    bad = assess_plausibility("1990-13-40", context=PlausibilityContext(today=dt.date(2005, 1, 1)))
    assert bad.ok is False
    assert bad.score == 0.0


def test_plausibility_rejects_year_before_minimum() -> None:
    ctx = PlausibilityContext(today=dt.date(2005, 1, 1), plausible_year_min=1970)
    res = assess_plausibility("1969-12-31", context=ctx)
    assert res.ok is False
    assert any("before plausible minimum" in p.lower() for p in res.penalties)
    assert res.score < 0.5


def test_plausibility_rejects_future_date() -> None:
    ctx = PlausibilityContext(today=dt.date(2000, 1, 1), plausible_year_max_future_years=0)
    res = assess_plausibility("2000-01-02", context=ctx)
    assert res.ok is False
    assert any("future" in p.lower() for p in res.penalties)
    assert res.score < 0.2


def test_plausibility_rejects_date_after_scan_date() -> None:
    ctx = PlausibilityContext(scan_date="1995-06-01", today=dt.date(2005, 1, 1))
    res = assess_plausibility("1995-06-02", context=ctx)
    assert res.ok is False
    assert any("after scan/file date" in p.lower() for p in res.penalties)
    assert res.score < 0.2


def test_plausibility_penalizes_outside_archive_observed_range() -> None:
    ctx = PlausibilityContext(
        today=dt.date(2010, 1, 1),
        scan_date="2010-01-01",
        archive_year_min=1982,
        archive_year_max=1990,
        archive_year_margin=2,
    )
    below = assess_plausibility("1979-01-01", context=ctx)  # below min-margin
    assert below.ok is False
    assert any("below observed archive range" in p.lower() for p in below.penalties)
    assert 0.0 < below.score < 1.0

    above = assess_plausibility("1995-01-01", context=ctx)  # above max+margin
    assert above.ok is False
    assert any("above observed archive range" in p.lower() for p in above.penalties)
    assert 0.0 < above.score < 1.0


def test_plausibility_accepts_boundary_cases() -> None:
    ctx = PlausibilityContext(
        today=dt.date(2000, 1, 10),
        scan_date="2000-01-10",
        plausible_year_min=1970,
        archive_year_min=1970,
        archive_year_max=1998,
        archive_year_margin=2,
    )
    at_min = assess_plausibility("1970-01-01", context=ctx)
    assert at_min.ok is True
    assert at_min.score > 0.9

    at_scan = assess_plausibility("2000-01-10", context=ctx)
    assert at_scan.ok is True

    at_archive_hi = assess_plausibility("2000-01-01", context=ctx)  # 2000 <= 1998+2
    assert at_archive_hi.ok is True


def test_plausibility_works_with_parse_result_date_string() -> None:
    parsed = parse_date("821224", field_order="YMD", auto_detect=False, pivot=29)
    assert parsed.parsed_date == "1982-12-24"
    ctx = PlausibilityContext(scan_date="2005-01-01", today=dt.date(2005, 1, 1))
    res = assess_plausibility(parsed.parsed_date, context=ctx)
    assert res.ok is True
    assert res.parsed_date == dt.date(1982, 12, 24)
