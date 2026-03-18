"""
temporal_security.py - Temporal validation / anti-leakage enforcement.

Core rule: A feature or context value is only point-in-time safe if its
KnowledgeUtc <= ObservationCutoffUtc.  If any component violates this,
the prediction still runs (for observability) but the sample is marked
ineligible for training memory.

This module is stateless — it inspects the payload and returns a verdict.
"""

from datetime import datetime, timezone


def _parse_utc(s: str | None) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp, returning None on failure."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def check_temporal_safety(payload: dict) -> dict:
    """
    Inspect the nested metabolic payload for temporal safety.

    Returns:
        {
            "passed": bool,
            "observation_cutoff_utc": str | None,
            "violations": [str, ...],
        }
    """
    violations: list[str] = []

    # Extract observation cutoff from temporal envelope
    temporal = payload.get("temporal") or payload.get("meta", {})
    cutoff_str = None
    if isinstance(temporal, dict):
        cutoff_str = temporal.get("observation_cutoff_utc")

    cutoff = _parse_utc(cutoff_str)

    if cutoff is None:
        # No cutoff means we can't verify — mark as unsafe
        return {
            "passed": False,
            "observation_cutoff_utc": cutoff_str,
            "violations": ["missing_observation_cutoff_utc"],
        }

    # Check macro section knowledge timestamps
    macro = payload.get("macro", {})
    if isinstance(macro, dict):
        _check_section_timestamp(macro, "cross_asset", cutoff, violations)
        _check_section_timestamp(macro, "regime_hints", cutoff, violations)
        _check_section_timestamp(macro, "scheduled", cutoff, violations)
        _check_section_timestamp(macro, "headlines", cutoff, violations)
        _check_section_timestamp(macro, "crypto_stress", cutoff, violations)

    # Check events section
    events = payload.get("events", {})
    if isinstance(events, dict):
        _check_section_timestamp(events, None, cutoff, violations)

    passed = len(violations) == 0
    return {
        "passed": passed,
        "observation_cutoff_utc": cutoff_str,
        "violations": violations,
    }


def _check_section_timestamp(
    section: dict,
    sub_key: str | None,
    cutoff: datetime,
    violations: list[str],
) -> None:
    """Check a section/sub-section for knowledge_utc > cutoff."""
    target = section
    label = "root"
    if sub_key is not None:
        target = section.get(sub_key, {})
        label = sub_key
        if not isinstance(target, dict):
            return

    knowledge_str = target.get("knowledge_utc") or target.get("as_of_utc")
    if knowledge_str is None:
        return  # no timestamp to check — not a violation by itself

    knowledge = _parse_utc(knowledge_str)
    if knowledge is None:
        return

    if knowledge > cutoff:
        violations.append(f"{label}_knowledge_after_cutoff")


def compute_eligibility(
    temporal_passed: bool,
    governance: dict | None = None,
) -> tuple[bool, list[str]]:
    """
    Determine training eligibility from temporal safety + governance signals.

    Returns (eligible: bool, block_reasons: [str])
    """
    reasons: list[str] = []

    if not temporal_passed:
        reasons.append("temporal_safety_failed")

    if governance and isinstance(governance, dict):
        if governance.get("breathless", False):
            reasons.append("system_breathless")
        if governance.get("overloaded", False):
            reasons.append("system_overloaded")
        if governance.get("learning_paused", False):
            reasons.append("learning_paused")

    eligible = len(reasons) == 0
    return eligible, reasons
