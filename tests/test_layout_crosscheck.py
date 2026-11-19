from __future__ import annotations

from datestamp.parse import ParseResult, check_layout_crosscheck


def test_layout_crosscheck_no_violations_for_consistent_ymd_batch() -> None:
    raws = ["821224", "830810", "900509", "990716"]  # YYMMDD
    report = check_layout_crosscheck(raws, assumed_field_order="YMD", pivot=29)
    assert report.assumed_field_order == "YMD"
    assert report.evaluated_items == 4
    assert report.violations == 0
    assert report.month_gt_12 == 0
    assert report.day_gt_31 == 0
    assert report.is_loud_violation is False
    assert "no field2/field3 bound violations" in report.message.lower() or "no" in report.message.lower()


def test_layout_crosscheck_detects_widespread_violations_when_order_wrong() -> None:
    # DMY-like digit strings evaluated under a wrong YMD assumption should violate field bounds.
    # e.g. 171288 -> (17,12,88) positional => field3 (day position in YMD assumption) = 88 > 31
    raws = ["171288", "121090", "251290", "11290", "4599"]
    report = check_layout_crosscheck(
        raws,
        assumed_field_order="YMD",
        pivot=29,
        loud_violation_threshold=0.2,
        trivial_violation_count=1,
    )
    assert report.evaluated_items == len(raws)
    assert report.violations >= 3
    assert (report.month_gt_12 + report.day_gt_31) >= 3
    assert report.is_loud_violation is True
    assert "LAYOUT WARNING" in report.message
    assert report.violation_rate >= 0.2


def test_layout_crosscheck_trivial_violation_not_loud() -> None:
    raws = ["821224", "830810", "900509", "171288"]  # one odd/inconsistent item
    report = check_layout_crosscheck(
        raws,
        assumed_field_order="YMD",
        pivot=29,
        loud_violation_threshold=0.5,
        trivial_violation_count=1,
    )
    assert report.violations >= 1
    assert report.is_loud_violation is False
    assert "minor/trivial" in report.message.lower() or "violation" in report.message.lower()


def test_layout_crosscheck_accepts_parse_results_and_dict_items() -> None:
    items = [
        ParseResult(raw_digits="821224", normalized_digits="821224", parsed_date="1982-12-24"),
        {"normalized_digits": "900509"},
        "830810",
    ]
    report = check_layout_crosscheck(items, assumed_field_order="YMD", pivot=29)
    assert report.total_items == 3
    assert report.evaluated_items == 3
    assert report.violations == 0


def test_layout_crosscheck_unsupported_order_returns_message() -> None:
    report = check_layout_crosscheck(["821224"], assumed_field_order="ABC")
    assert report.assumed_field_order == "ABC"
    assert "unsupported" in report.message.lower()
