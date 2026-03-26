"""
schema_gate.py — Compatibility gate preventing schema poisoning during training/inference.

Before any features are consumed by a model (controlled_fit, prediction, or
resolved sample loading), they pass through this gate.  The gate checks
whether the memory's feature schema matches the model's expectation and,
when possible, adapts mismatched memories via zero-padding or truncation.

Adapter contracts:
  v1.0.0 -> v2.0.0  :  zero-pad 20 macro feature slots (lossless)
  v2.0.0 -> v1.0.0  :  truncate to first 18 features   (lossy but safe)
  same    -> same    :  pass-through

Every adapted memory receives a provenance tag so downstream consumers
know the features have been transformed.

Usage sites:
  - controlled_fit (training gate)
  - resolved sample loading (label_resolver / pending_memory)
  - challenger evaluation (offline comparison)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from .feature_adapter import FEATURE_NAMES_V1, FEATURE_NAMES_V2_MACRO, FEATURE_NAMES
from .model_registry import (
    FEATURE_SCHEMA_REGISTRY,
    check_schema_compatibility,
)


# ═══════════════════════════════════════════════════════════════════
# SCHEMA COMPATIBILITY RESULT
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SchemaCompatibilityResult:
    """
    Outcome of a schema compatibility check.

    Fields:
      compatible        — True if the memory can be used (directly or via adapter)
      reason            — human-readable explanation of the decision
      adapter_available — True if an adapter can bridge the schema gap
      source_version    — the memory's feature version
      target_version    — the model's expected feature version
    """

    compatible: bool
    reason: str
    adapter_available: bool
    source_version: str
    target_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "reason": self.reason,
            "adapter_available": self.adapter_available,
            "source_version": self.source_version,
            "target_version": self.target_version,
        }


# ═══════════════════════════════════════════════════════════════════
# COMPATIBILITY CHECKING
# ═══════════════════════════════════════════════════════════════════

def check_memory_compatibility(
    memory_feature_version: str,
    model_feature_version: str,
) -> SchemaCompatibilityResult:
    """
    Check whether a memory's features are compatible with a model's expectations.

    Delegates to the model_registry's schema compatibility engine, then
    wraps the result in a typed SchemaCompatibilityResult.

    Args:
      memory_feature_version — feature schema version stored with the memory
      model_feature_version  — feature schema version the model expects

    Returns:
      SchemaCompatibilityResult with compatibility verdict and adapter info
    """
    raw = check_schema_compatibility(memory_feature_version, model_feature_version)
    return SchemaCompatibilityResult(
        compatible=raw["compatible"],
        reason=raw["reason"],
        adapter_available=raw["adapter_available"],
        source_version=raw["source_version"],
        target_version=raw["target_version"],
    )


# ═══════════════════════════════════════════════════════════════════
# FEATURE ADAPTATION
# ═══════════════════════════════════════════════════════════════════

# Known adapter identifiers
_ADAPTER_V1_TO_V2 = "zero_pad_v1_to_v2"
_ADAPTER_V2_TO_V1 = "truncate_v2_to_v1"

# Expected feature counts per version
_V1_FEATURE_COUNT = len(FEATURE_NAMES_V1)       # 18
_V2_FEATURE_COUNT = len(FEATURE_NAMES)           # 38
_MACRO_PAD_COUNT = len(FEATURE_NAMES_V2_MACRO)   # 20


def adapt_features(
    features: list[float],
    source_version: str,
    target_version: str,
) -> list[float]:
    """
    Adapt a feature vector from one schema version to another.

    Supported transitions:
      v1.0.0 -> v2.0.0  :  zero-pad 20 macro slots (lossless)
      v2.0.0 -> v1.0.0  :  truncate to first 18 (lossy but safe)
      same   -> same     :  pass-through

    Args:
      features       — the raw feature vector from the memory
      source_version — the version the features were extracted under
      target_version — the version the consuming model expects

    Returns:
      Adapted feature vector with the target dimensionality.

    Raises:
      ValueError — if no adapter exists for the requested transition
      ValueError — if the source vector has an unexpected length
    """
    # Pass-through: same version
    if source_version == target_version:
        return list(features)

    # v1 -> v2: zero-pad macro features
    if source_version == "v1.0.0" and target_version == "v2.0.0":
        if len(features) < _V1_FEATURE_COUNT:
            print(
                f"[SchemaGate] WARN: v1 vector has {len(features)} features, "
                f"expected {_V1_FEATURE_COUNT}; padding source first",
                file=sys.stderr,
            )
            features = list(features) + [0.0] * (_V1_FEATURE_COUNT - len(features))
        elif len(features) > _V1_FEATURE_COUNT:
            print(
                f"[SchemaGate] WARN: v1 vector has {len(features)} features, "
                f"expected {_V1_FEATURE_COUNT}; truncating source to v1 length",
                file=sys.stderr,
            )
            features = features[:_V1_FEATURE_COUNT]

        # Append zeros for the 20 macro feature slots
        return list(features[:_V1_FEATURE_COUNT]) + [0.0] * _MACRO_PAD_COUNT

    # v2 -> v1: truncate to technical features only
    if source_version == "v2.0.0" and target_version == "v1.0.0":
        if len(features) < _V1_FEATURE_COUNT:
            print(
                f"[SchemaGate] WARN: v2 vector too short ({len(features)}), "
                f"padding to v1 length",
                file=sys.stderr,
            )
            result = list(features) + [0.0] * (_V1_FEATURE_COUNT - len(features))
            return result[:_V1_FEATURE_COUNT]

        return list(features[:_V1_FEATURE_COUNT])

    # No adapter available for this transition
    raise ValueError(
        f"No adapter available for schema transition "
        f"'{source_version}' -> '{target_version}'. "
        f"Known adapters: v1.0.0->v2.0.0, v2.0.0->v1.0.0"
    )


def _get_adapter_name(source_version: str, target_version: str) -> str:
    """Return the canonical adapter identifier for a version transition."""
    if source_version == "v1.0.0" and target_version == "v2.0.0":
        return _ADAPTER_V1_TO_V2
    if source_version == "v2.0.0" and target_version == "v1.0.0":
        return _ADAPTER_V2_TO_V1
    return f"unknown_{source_version}_to_{target_version}"


def _build_provenance_tag(
    source_version: str,
    target_version: str,
) -> dict[str, str]:
    """
    Build a provenance tag for an adapted memory.

    The tag records what transformation was applied so downstream consumers
    know the features are not in their original form.
    """
    return {
        "adapted_from": source_version,
        "adapted_to": target_version,
        "adapter": _get_adapter_name(source_version, target_version),
    }


# ═══════════════════════════════════════════════════════════════════
# MEMORY FILTERING
# ═══════════════════════════════════════════════════════════════════

def filter_compatible_memories(
    memories: list[dict[str, Any]],
    model_feature_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Partition a list of memories into three buckets based on schema compatibility
    with the target model.

    Each memory dict is expected to have a "feature_version" key. Memories
    without this key are treated as having the model's own version (legacy
    backward-compat assumption).

    Args:
      memories              — list of memory dicts (resolved samples)
      model_feature_version — the feature version the consuming model expects

    Returns:
      A 3-tuple of (compatible, incompatible, adapted):
        compatible   — memories whose feature_version matches exactly
        incompatible — memories that cannot be used (no adapter)
        adapted      — memories that were transformed via an adapter,
                       each with an injected "schema_provenance" tag
    """
    compatible: list[dict[str, Any]] = []
    incompatible: list[dict[str, Any]] = []
    adapted: list[dict[str, Any]] = []

    adapt_ok = 0
    adapt_fail = 0

    for memory in memories:
        mem_version = memory.get("feature_version", model_feature_version)

        # Exact match: no adaptation needed
        if mem_version == model_feature_version:
            compatible.append(memory)
            continue

        # Check compatibility
        compat = check_memory_compatibility(mem_version, model_feature_version)

        if not compat.compatible:
            incompatible.append(memory)
            continue

        if not compat.adapter_available:
            # Compatible but no adapter (shouldn't happen with current schemas,
            # but defensive coding for future schema additions)
            incompatible.append(memory)
            continue

        # Attempt adaptation
        features = memory.get("features")
        if features is None or not isinstance(features, (list, tuple)):
            incompatible.append(memory)
            adapt_fail += 1
            continue

        try:
            adapted_features = adapt_features(
                list(features), mem_version, model_feature_version
            )
        except (ValueError, TypeError) as ex:
            print(
                f"[SchemaGate] Adaptation failed for memory "
                f"({mem_version}->{model_feature_version}): {ex}",
                file=sys.stderr,
            )
            incompatible.append(memory)
            adapt_fail += 1
            continue

        # Build adapted copy with provenance
        adapted_memory = dict(memory)
        adapted_memory["features"] = adapted_features
        adapted_memory["feature_version"] = model_feature_version
        adapted_memory["schema_provenance"] = _build_provenance_tag(
            mem_version, model_feature_version
        )
        adapted.append(adapted_memory)
        adapt_ok += 1

    # Diagnostic summary
    total = len(memories)
    if total > 0:
        print(
            f"[SchemaGate] Filtered {total} memories for {model_feature_version}: "
            f"{len(compatible)} direct, {len(adapted)} adapted, "
            f"{len(incompatible)} incompatible"
            + (f" ({adapt_fail} adapt failures)" if adapt_fail else ""),
            file=sys.stderr,
        )

    return compatible, incompatible, adapted
