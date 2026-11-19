"""Date parsing, field-order handling, century pivot, and layout cross-check.

This module is photo-free and deterministic. It normalizes raw digit readings,
parses candidate date formats (including variable-length/ambiguous printed stamps),
infers a two-digit-year pivot from batch distributions, and reports batch layout
cross-check diagnostics.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence


_FIELD_ORDERS = ("YMD", "MDY", "DMY")


@dataclass
class ParseResult:
    """Result of parsing a single raw stamp reading."""

    raw_digits: str
    normalized_digits: str
    parsed_date: str = ""  # YYYY-MM-DD or empty on failure
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    field_order: str = ""  # effective/inferred order used (Y/M/D letters)
    pivot_used: Optional[int] = None
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.parsed_date)


@dataclass
class PivotInference:
    pivot: int
    rationale: str
    years_seen: list[int] = field(default_factory=list)
    expanded_years: list[int] = field(default_factory=list)


@dataclass
class LayoutCrosscheckReport:
    assumed_field_order: str
    total_items: int = 0
    evaluated_items: int = 0
    month_gt_12: int = 0
    day_gt_31: int = 0
    invalid_calendar: int = 0
    violations: int = 0
    violation_rate: float = 0.0
    is_loud_violation: bool = False
    threshold: float = 0.1
    message: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumed_field_order": self.assumed_field_order,
            "total_items": self.total_items,
            "evaluated_items": self.evaluated_items,
            "month_gt_12": self.month_gt_12,
            "day_gt_31": self.day_gt_31,
            "invalid_calendar": self.invalid_calendar,
            "violations": self.violations,
            "violation_rate": self.violation_rate,
            "is_loud_violation": self.is_loud_violation,
            "threshold": self.threshold,
            "message": self.message,
            "examples": self.examples,
        }


def normalize_digits(raw: str | None) -> str:
    """Normalize a raw reading by extracting digit characters in order.

    Separators/spaces are removed. Non-digit markers (e.g. '?') are not preserved
    as digits; callers should treat presence of '?' as ambiguity/unreadable.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    return "".join(ch for ch in s if ch.isdigit())


def has_unreadable_marker(raw: str | None) -> bool:
    if raw is None:
        return False
    return "?" in str(raw)


def _is_valid_calendar_date(year: int, month: int, day: int) -> bool:
    if year < 1 or month < 1 or month > 12 or day < 1 or day > 31:
        return False
    try:
        _dt.date(year, month, day)
        return True
    except Exception:
        return False


def expand_two_digit_year(yy: int, pivot: int) -> int:
    """Expand a two-digit year using an inclusive 2000s cutoff pivot.

    `pivot` is interpreted as the maximum YY mapped to 20YY. Values above pivot
    map to 19YY. Example: pivot=29 => 00-29->2000-2029, 30-99->1930-1999.
    """
    if yy < 0 or yy > 99:
        raise ValueError(f"two-digit year out of range: {yy}")
    if pivot < 0 or pivot > 99:
        raise ValueError(f"pivot out of range: {pivot}")
    return (2000 + yy) if yy <= pivot else (1900 + yy)


def infer_century_pivot(
    two_digit_years: Sequence[int],
    *,
    default_pivot: int = 29,
    min_year: int = 1970,
    max_year: int = 2039,
) -> PivotInference:
    """Infer a pivot (max YY -> 20YY) from observed two-digit years.

    Heuristic: choose the pivot in [0..99] that yields expanded years within
    [min_year, max_year] for as many observations as possible; break ties by
    preferring a compact year range and then by closeness to default_pivot.
    """
    cleaned = [int(y) for y in two_digit_years if y is not None and 0 <= int(y) <= 99]
    if not cleaned:
        return PivotInference(
            pivot=default_pivot,
            rationale=f"No two-digit years observed; using default pivot {default_pivot}.",
            years_seen=[],
            expanded_years=[],
        )

    best: Optional[tuple] = None
    best_pivot = default_pivot
    best_expanded: list[int] = []
    for pivot in range(0, 100):
        expanded = [expand_two_digit_year(yy, pivot) for yy in cleaned]
        in_range = sum(1 for y in expanded if min_year <= y <= max_year)
        span = (max(expanded) - min(expanded)) if expanded else 10**9
        # prefer more in-range, then plausible compact span, then smaller distance to default
        key = (-in_range, span, abs(pivot - default_pivot), pivot)
        if best is None or key < best:
            best = key
            best_pivot = pivot
            best_expanded = expanded

    in_range = sum(1 for y in best_expanded if min_year <= y <= max_year)
    yr_min = min(best_expanded) if best_expanded else None
    yr_max = max(best_expanded) if best_expanded else None
    rationale = (
        f"Inferred pivot {best_pivot} (YY<={best_pivot}->20YY, else 19YY) from "
        f"{len(cleaned)} year samples; {in_range}/{len(cleaned)} expanded years in "
        f"[{min_year},{max_year}], range {yr_min}-{yr_max}."
    )
    return PivotInference(
        pivot=best_pivot,
        rationale=rationale,
        years_seen=cleaned,
        expanded_years=best_expanded,
    )


def infer_batch_pivot(
    raw_digit_strings: Sequence[str],
    *,
    assumed_field_order: str = "YMD",
    default_pivot: int = 29,
) -> PivotInference:
    """Infer pivot from a batch of raw digit strings using best-effort extraction.

    This uses `parse_date(..., auto_detect=True)` with a temporary pivot guess
    only to extract plausible two-digit years for inference, then returns the
    inferred pivot. Callers should re-parse with the returned pivot for final
    results.
    """
    years: list[int] = []
    for raw in raw_digit_strings:
        if has_unreadable_marker(raw):
            continue
        norm = normalize_digits(raw)
        if len(norm) < 4:
            continue
        # try parse with default pivot to pick a likely candidate/year token
        res = parse_date(norm, field_order=assumed_field_order, pivot=default_pivot, auto_detect=True)
        if res.ok and res.year is not None:
            yy = res.year % 100
            years.append(yy)
        else:
            # fallback heuristics: common 6-digit forms and 4/5 endings
            if len(norm) == 6:
                years.extend([int(norm[0:2]), int(norm[4:6])])
            elif len(norm) >= 4:
                years.append(int(norm[-2:]))
    return infer_century_pivot(years, default_pivot=default_pivot)


def _split_candidates_for_order(digits: str, order: str) -> list[tuple[int, int, int, str]]:
    """Return possible (year_val_or_yy, month, day, note) component candidates.

    For two-digit year interpretations, year is returned as 0-99. For 4-digit
    years, 1000-2999. Components are not calendar-validated here except basic ranges.
    """
    order = order.upper()
    if order not in _FIELD_ORDERS:
        return []
    n = len(digits)
    if not digits.isdigit() or n < 4 or n > 8:
        return []

    out: list[tuple[int, int, int, str]] = []

    def add(y_raw: str, m_raw: str, d_raw: str, note: str) -> None:
        if not (y_raw and m_raw and d_raw):
            return
        if any(len(x) == 0 for x in (y_raw, m_raw, d_raw)):
            return
        # reject empty or absurd lengths
        if len(m_raw) > 2 or len(d_raw) > 2 or len(y_raw) not in (1, 2, 3, 4):
            return
        try:
            yv = int(y_raw)
            mv = int(m_raw)
            dv = int(d_raw)
        except Exception:
            return
        if len(y_raw) == 4:
            if yv < 1000 or yv > 2999:
                return
            year_token = yv
        else:
            # 1-3 digit year tokens treated as two-digit year if possible by padding semantics
            # but printed stamps are expected to be 2-digit years; 1-digit is ambiguous and low confidence.
            if len(y_raw) == 3:
                return
            if len(y_raw) == 1:
                # interpret as 200x/190x? too ambiguous, skip
                return
            year_token = yv  # 0-99
        if mv <= 0 or dv <= 0:
            return
        out.append((year_token, mv, dv, note))

    # 6-digit unambiguous-by-length common forms
    if n == 6:
        a, b, c = digits[0:2], digits[2:4], digits[4:6]
        mapping = {"Y": a, "M": b, "D": c} if order == "YMD" else ({"M": a, "D": b, "Y": c} if order == "MDY" else {"D": a, "M": b, "Y": c})
        add(mapping["Y"], mapping["M"], mapping["D"], "split:2-2-2")
        return out

    # 8-digit forms (YYYYMMDD etc.)
    if n == 8:
        if order == "YMD":
            add(digits[0:4], digits[4:6], digits[6:8], "split:4-2-2")
        elif order == "MDY":
            add(digits[4:8], digits[0:2], digits[2:4], "split:2-2-4")
        elif order == "DMY":
            add(digits[4:8], digits[2:4], digits[0:2], "split:2-2-4")
        return out

    # 4-5-7: enumerate partitions into 3 fields (Y/M/D lengths depend on order)
    # Generate all length combinations summing to n with reasonable per-field lengths.
    # Y:1-4, M:1-2, D:1-2
    def lens_for(field: str) -> range:
        if field == "Y":
            return range(1, 5)
        return range(1, 3)

    fields = list(order)
    l0s = lens_for(fields[0])
    l1s = lens_for(fields[1])
    l2s = lens_for(fields[2])
    for l0 in l0s:
        for l1 in l1s:
            l2 = n - l0 - l1
            if l2 not in l2s:
                continue
            p0 = digits[0:l0]
            p1 = digits[l0 : l0 + l1]
            p2 = digits[l0 + l1 : l0 + l1 + l2]
            parts = {fields[0]: p0, fields[1]: p1, fields[2]: p2}
            add(parts["Y"], parts["M"], parts["D"], f"split:{l0}-{l1}-{l2}")
    # de-dup while preserving order
    seen = set()
    uniq = []
    for y, m, d, note in out:
        key = (y, m, d, note)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


def _score_candidate(
    year: int,
    month: int,
    day: int,
    *,
    order: str,
    normalized_digits: str,
    note: str,
    preferred_order: Optional[str],
    pivot: int,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if not _is_valid_calendar_date(year, month, day):
        return (-1.0, ["invalid calendar date"])
    score += 2.0
    reasons.append("valid calendar date")

    # range plausibility (family archive scanned later; rough defaults)
    if 1970 <= year <= 2015:
        score += 1.0
        reasons.append("year in plausible archive range")
    elif 1950 <= year <= 2035:
        score += 0.4
        reasons.append("year in extended plausible range")
    else:
        score -= 0.8
        reasons.append("year outside plausible range")

    # prefer explicit 6-digit full forms
    if len(normalized_digits) == 6 and note.startswith("split:2-2-2"):
        score += 0.25
    # prefer two-digit month/day tokens when unambiguous length allows? minor
    if len(normalized_digits) in (4, 5):
        score += 0.05

    if preferred_order and order == preferred_order.upper():
        score += 0.2
        reasons.append("matches preferred field order")

    # heuristic: common stamp formats in this domain are YMD or DMY; slight tie-break
    if order in ("YMD", "DMY"):
        score += 0.02

    # mild preference for pivots near inferred/default behavior already encoded in year choice
    _ = pivot
    return (score, reasons)


def parse_date(
    raw_digits: str | None,
    *,
    field_order: str = "YMD",
    pivot: Optional[int] = None,
    auto_detect: Optional[bool] = None,
    default_pivot: int = 29,
) -> ParseResult:
    """Parse a raw stamp reading into a calendar date.

    - `field_order`: preferred/assumed order (YMD/MDY/DMY). Used directly unless
      `auto_detect` selects a better candidate.
    - `auto_detect`: when True, try all supported orders and pick the best valid
      candidate. When False, only try `field_order`. When None (default), auto-detect
      is enabled for 4-5 digit ambiguous inputs and when field_order is YMD-like
      uncertainty exists? we default to True for robustness.
    - `pivot`: max YY mapped to 20YY for two-digit years. If None, `default_pivot`
      is used (batch workflows should infer and pass an explicit pivot).
    """
    raw_in = "" if raw_digits is None else str(raw_digits)
    res = ParseResult(raw_digits=raw_in, normalized_digits=normalize_digits(raw_in))
    if has_unreadable_marker(raw_in):
        res.notes.append("raw digits contain unreadable marker '?'")
        return res
    norm = res.normalized_digits
    if not norm:
        res.notes.append("no digits to parse")
        return res
    if len(norm) < 4:
        res.notes.append(f"too few digits to parse ({len(norm)})")
        return res

    fo = (field_order or "YMD").upper()
    if fo not in _FIELD_ORDERS:
        res.notes.append(f"unsupported field_order '{field_order}', falling back to YMD")
        fo = "YMD"
    if pivot is None:
        pivot = default_pivot
    res.pivot_used = pivot

    if auto_detect is None:
        auto_detect = True

    orders_to_try = list(_FIELD_ORDERS) if auto_detect else [fo]
    # keep preferred order first for stable tie-breaking context
    if fo in orders_to_try:
        orders_to_try.remove(fo)
        orders_to_try.insert(0, fo)

    candidates: list[dict[str, Any]] = []
    for order in orders_to_try:
        for year_token, month, day, split_note in _split_candidates_for_order(norm, order):
            if year_token <= 99:
                year = expand_two_digit_year(int(year_token), pivot)
                yy_note = f"yy={int(year_token):02d}->yyyy={year}"
            else:
                year = int(year_token)
                yy_note = f"yyyy={year}"
            score, reasons = _score_candidate(
                year,
                month,
                day,
                order=order,
                normalized_digits=norm,
                note=split_note,
                preferred_order=fo,
                pivot=pivot,
            )
            if score < 0:
                continue
            candidates.append(
                {
                    "parsed_date": f"{year:04d}-{month:02d}-{day:02d}",
                    "year": year,
                    "month": month,
                    "day": day,
                    "field_order": order,
                    "score": score,
                    "split": split_note,
                    "year_note": yy_note,
                    "reasons": reasons,
                }
            )

    # de-duplicate candidates by parsed date + order, keep best score
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for c in candidates:
        key = (c["parsed_date"], c["field_order"])
        if key not in dedup or c["score"] > dedup[key]["score"]:
            dedup[key] = c
    ranked = sorted(
        dedup.values(),
        key=lambda c: (
            -float(c["score"]),
            0 if c["field_order"] == fo else 1,
            c["parsed_date"],
        ),
    )
    res.candidates = ranked

    if not ranked:
        res.notes.append("no valid calendar date candidates found")
        if len(norm) not in (4, 5, 6, 7, 8):
            res.notes.append(f"unsupported digit length {len(norm)}")
        return res

    best = ranked[0]
    # detect ambiguity: another different date close in score
    ambiguous = False
    if len(ranked) > 1:
        second = ranked[1]
        if second["parsed_date"] != best["parsed_date"] and (best["score"] - second["score"]) < 0.15:
            ambiguous = True
            res.notes.append(
                f"ambiguous parse: {best['parsed_date']} ({best['field_order']}) vs {second['parsed_date']} ({second['field_order']})"
            )

    res.parsed_date = best["parsed_date"]
    res.year = int(best["year"])
    res.month = int(best["month"])
    res.day = int(best["day"])
    res.field_order = str(best["field_order"])
    # normalize confidence into [0,1] roughly
    conf = min(1.0, max(0.0, (float(best["score"]) / 3.3)))
    if ambiguous:
        conf *= 0.7
    res.confidence = conf
    res.notes.append(f"selected {best['field_order']} via {best['split']} ({best['year_note']})")
    if auto_detect and res.field_order != fo:
        res.notes.append(f"inferred field order {res.field_order} (preferred {fo})")
    return res


def parse_dates_for_batch(
    raw_digit_strings: Sequence[str],
    *,
    field_order: str = "YMD",
    pivot: Optional[int] = None,
    auto_detect: bool = True,
) -> tuple[list[ParseResult], PivotInference]:
    """Parse a batch with inferred pivot (if not provided) and return results+pivot info."""
    if pivot is None:
        pivot_info = infer_batch_pivot(raw_digit_strings, assumed_field_order=field_order)
        pivot = pivot_info.pivot
    else:
        pivot_info = PivotInference(pivot=pivot, rationale=f"Using provided pivot {pivot}.")
    results = [
        parse_date(raw, field_order=field_order, pivot=pivot, auto_detect=auto_detect)
        for raw in raw_digit_strings
    ]
    return results, pivot_info


def _extract_field_values_for_order(
    normalized_digits: str,
    assumed_field_order: str,
    *,
    pivot: Optional[int] = None,
) -> Optional[tuple[int, int, int]]:
    """Best-effort extraction of numeric (field1, field2, field3) for assumed order.

    Returns component values in the assumed field order positions (Y/M/D values in
    that positional sequence), or None if not extractable. Used for layout checks.
    """
    fo = (assumed_field_order or "YMD").upper()
    if fo not in _FIELD_ORDERS or len(normalized_digits) < 4:
        return None
    # Try to parse preferring assumed order (no auto-detect) to evaluate that layout assumption
    res = parse_date(normalized_digits, field_order=fo, pivot=pivot, auto_detect=False)
    if res.ok and res.field_order == fo and res.year is not None and res.month is not None and res.day is not None:
        comp = {"Y": res.year % 100 if res.year >= 1900 else res.year, "M": res.month, "D": res.day}
        return (int(comp[fo[0]]), int(comp[fo[1]]), int(comp[fo[2]]))
    # fallback: try direct split candidates for assumed order only
    cands = _split_candidates_for_order(normalized_digits, fo)
    if not cands:
        return None
    # pick first plausible-ish candidate (calendar-agnostic for layout counting except basic values)
    ytok, m, d, _note = cands[0]
    yv = ytok % 100 if ytok >= 100 else ytok
    comp = {"Y": int(yv), "M": int(m), "D": int(d)}
    return (int(comp[fo[0]]), int(comp[fo[1]]), int(comp[fo[2]]))


def check_layout_crosscheck(
    items: Sequence[Any],
    assumed_field_order: str = "YMD",
    *,
    pivot: Optional[int] = None,
    loud_violation_threshold: float = 0.1,
    trivial_violation_count: int = 1,
) -> LayoutCrosscheckReport:
    """Check pooled layout consistency for a batch.

    `items` may be raw digit strings or ParseResult-like objects with
    `normalized_digits`/`raw_digits`.

    Per spec (safety net): for the assumed layout, field 2 should never exceed 12
    and field 3 should never exceed 31. If violated in more than a trivial number
    of photos, report loudly (layout assumption likely wrong for part of archive).
    """
    fo = (assumed_field_order or "YMD").upper()
    report = LayoutCrosscheckReport(assumed_field_order=fo, threshold=loud_violation_threshold)
    if fo not in _FIELD_ORDERS:
        report.message = f"Unsupported assumed field order '{assumed_field_order}'."
        return report

    # Identify which positions correspond to month/day under assumed order
    month_pos = fo.index("M")  # 0-based among (f1,f2,f3)
    day_pos = fo.index("D")

    for item in items:
        report.total_items += 1
        if isinstance(item, ParseResult):
            norm = item.normalized_digits or normalize_digits(item.raw_digits)
        elif isinstance(item, dict):
            norm = normalize_digits(item.get("normalized_digits") or item.get("raw_digits") or item.get("digits") or "")
        else:
            norm = normalize_digits(str(item))
        if not norm or len(norm) < 4:
            continue
        vals = _extract_field_values_for_order(norm, fo, pivot=pivot)
        if vals is None:
            continue
        report.evaluated_items += 1
        f1, f2, f3 = vals
        fields = [f1, f2, f3]
        month_val = int(fields[month_pos])
        day_val = int(fields[day_pos])

        item_viol = False
        reasons = []
        # Spec-stated checks: field 2 <= 12 and field 3 <= 31 in assumed layout positions (1-based in text)
        if int(fields[1]) > 12:
            report.month_gt_12 += 1
            item_viol = True
            reasons.append(f"field2={int(fields[1])}>12")
        if int(fields[2]) > 31:
            report.day_gt_31 += 1
            item_viol = True
            reasons.append(f"field3={int(fields[2])}>31")

        # semantic checks by role (useful diagnostics; may overlap with spec checks)
        if month_val > 12 and month_pos != 1:
            reasons.append(f"month(value={month_val})>12")
            item_viol = True
        if day_val > 31 and day_pos != 2:
            reasons.append(f"day(value={day_val})>31")
            item_viol = True

        # invalid calendar for extracted Y/M/D if we can form full year
        comp = {fo[i]: fields[i] for i in range(3)}
        yy = int(comp["Y"])
        mm = int(comp["M"])
        dd = int(comp["D"])
        yyyy = expand_two_digit_year(yy % 100, pivot if pivot is not None else 29) if yy < 100 else yy
        if mm > 12 or dd > 31 or not _is_valid_calendar_date(yyyy, mm, dd):
            report.invalid_calendar += 1

        if item_viol:
            report.violations += 1
            if len(report.examples) < 10:
                report.examples.append(
                    {
                        "normalized_digits": norm,
                        "assumed_field_order": fo,
                        "fields_positional": [f1, f2, f3],
                        "month": mm,
                        "day": dd,
                        "year_component": yy,
                        "reasons": reasons,
                    }
                )

    if report.evaluated_items > 0:
        report.violation_rate = report.violations / float(report.evaluated_items)
    # "more than a trivial number of photos" -> either count threshold or rate threshold
    report.is_loud_violation = report.violations > trivial_violation_count and report.violation_rate >= loud_violation_threshold
    if report.is_loud_violation:
        report.message = (
            f"LAYOUT WARNING: assumed field order {fo} shows {report.violations}/"
            f"{report.evaluated_items} violating items ({report.violation_rate:.1%}). "
            f"Field2>12: {report.month_gt_12}, Field3>31: {report.day_gt_31}. "
            f"Layout assumption may be wrong for part of the archive; consider --field-order."
        )
    elif report.violations > 0:
        report.message = (
            f"Layout cross-check ({fo}): {report.violations}/{report.evaluated_items} minor/trivial violations "
            f"(Field2>12: {report.month_gt_12}, Field3>31: {report.day_gt_31})."
        )
    else:
        report.message = f"Layout cross-check ({fo}): no field2/field3 bound violations in {report.evaluated_items} evaluated items."
    return report

