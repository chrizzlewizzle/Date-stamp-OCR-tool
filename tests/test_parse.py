from __future__ import annotations

from datestamp.parse import normalize_digits, parse_date, parse_dates_for_batch


def test_normalize_digits_strips_separators_and_keeps_order() -> None:
    assert normalize_digits("82-12-24") == "821224"
    assert normalize_digits(" 12/10/90 ") == "121090"
    assert normalize_digits("ab 90 6 1 cd") == "9061"
    assert normalize_digits("") == ""
    assert normalize_digits(None) == ""


def test_parse_unreadable_marker_and_short_input() -> None:
    r = parse_date("99?16")
    assert r.parsed_date == ""
    assert r.normalized_digits == "9916"
    assert any("unreadable" in n.lower() or "?" in n for n in r.notes)

    r2 = parse_date("123")
    assert r2.parsed_date == ""
    assert any("too few" in n.lower() for n in r2.notes)


def test_parse_six_digit_formats_by_field_order() -> None:
    # YYMMDD
    r = parse_date("821224", field_order="YMD", auto_detect=False, pivot=29)
    assert r.parsed_date == "1982-12-24"
    assert r.field_order == "YMD"

    # MMDDYY
    r = parse_date("122482", field_order="MDY", auto_detect=False, pivot=29)
    assert r.parsed_date == "1982-12-24"
    assert r.field_order == "MDY"

    # DDMMYY
    r = parse_date("241282", field_order="DMY", auto_detect=False, pivot=29)
    assert r.parsed_date == "1982-12-24"
    assert r.field_order == "DMY"


def test_parse_variable_length_ground_truth_like_patterns_with_auto_detect() -> None:
    # Digit-string patterns similar to samples; auto-detect should resolve these cases.
    cases = [
        ("99716", "1999-07-16"),
        ("4599", "1999-05-04"),
        ("82318", "1982-03-18"),
        ("8249", "1982-04-09"),
        ("821224", "1982-12-24"),
        ("9059", "1990-05-09"),
        ("83810", "1983-08-10"),
        ("83214", "1983-02-14"),
        ("83219", "1983-02-19"),
        ("83218", "1983-02-18"),
        ("9061", "1990-06-01"),
        ("61090", "1990-10-06"),
        ("171288", "1988-12-17"),
        ("121090", "1990-10-12"),
    ]
    for raw, expected in cases:
        r = parse_date(raw, field_order="YMD", pivot=29, auto_detect=True)
        assert r.parsed_date == expected, f"{raw} -> {r.parsed_date}, notes={r.notes}, candidates={r.candidates[:3]}"


def test_parse_field_order_override_can_force_preferred_interpretation() -> None:
    # "010203" is ambiguous; with explicit non-auto-detect order it should follow that order
    r_ymd = parse_date("010203", field_order="YMD", auto_detect=False, pivot=29)
    assert r_ymd.parsed_date == "2001-02-03"
    r_mdy = parse_date("010203", field_order="MDY", auto_detect=False, pivot=29)
    assert r_mdy.parsed_date == "2003-01-02"
    r_dmy = parse_date("010203", field_order="DMY", auto_detect=False, pivot=29)
    assert r_dmy.parsed_date == "2003-02-01"


def test_parse_rejects_invalid_calendar_candidates() -> None:
    r = parse_date("991332", field_order="YMD", auto_detect=False, pivot=29)  # month 13, day 32
    assert r.parsed_date == ""
    assert any("no valid" in n.lower() for n in r.notes)


def test_parse_dates_for_batch_infers_pivot_and_parses() -> None:
    raws = ["821224", "9059", "99716", "171288"]
    results, pivot_info = parse_dates_for_batch(raws, field_order="YMD", auto_detect=True)
    assert pivot_info.pivot >= 0
    assert all(r.ok for r in results)
    assert [r.parsed_date for r in results] == [
        "1982-12-24",
        "1990-05-09",
        "1999-07-16",
        "1988-12-17",
    ]
    assert "pivot" in pivot_info.rationale.lower()
