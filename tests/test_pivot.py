from __future__ import annotations

from datestamp.parse import expand_two_digit_year, infer_batch_pivot, infer_century_pivot


def test_expand_two_digit_year_with_pivot() -> None:
    assert expand_two_digit_year(5, 29) == 2005
    assert expand_two_digit_year(29, 29) == 2029
    assert expand_two_digit_year(30, 29) == 1930
    assert expand_two_digit_year(99, 29) == 1999
    assert expand_two_digit_year(0, 5) == 2000
    assert expand_two_digit_year(82, 50) == 1982


def test_infer_century_pivot_defaults_when_no_data() -> None:
    info = infer_century_pivot([], default_pivot=29)
    assert info.pivot == 29
    assert info.years_seen == []
    assert "default" in info.rationale.lower()


def test_infer_century_pivot_keeps_archive_in_1900s_for_typical_samples() -> None:
    # Typical sample years 82,83,86,88,90,99 should map to 1982-1999
    years = [82, 83, 86, 88, 90, 99, 99, 82]
    info = infer_century_pivot(years, default_pivot=29)
    assert all(1980 <= y <= 1999 for y in info.expanded_years)
    # pivot 29 is sufficient/consistent (>=? 99 must map to 19xx so pivot <= 98)
    assert info.pivot <= 98


def test_infer_century_pivot_allows_plausible_2000s_when_mixed() -> None:
    # Mixed archive crossing century: 98,99,01,03 should prefer 1998,1999,2001,2003
    years = [98, 99, 1, 3, 2]
    info = infer_century_pivot(years, default_pivot=29)
    assert set(info.expanded_years) == {1998, 1999, 2001, 2003, 2002}
    assert info.pivot >= 3 and info.pivot < 98


def test_infer_batch_pivot_from_raw_digit_strings() -> None:
    raws = ["821224", "9059", "99716", "171288", "121090"]
    info = infer_batch_pivot(raws, assumed_field_order="YMD", default_pivot=29)
    assert all(1980 <= y <= 2005 for y in info.expanded_years)
    assert info.pivot >= 0
    assert "inferred pivot" in info.rationale.lower()
