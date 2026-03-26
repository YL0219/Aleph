"""
model_registry.py — Identity system for ML models in a multi-model future.

Tracks model identities, their feature contracts, schema compatibility,
and lifecycle status.  The registry is the single source of truth for
which models exist, which are active, and what feature schema they expect.

Storage layout:
  data_lake/cortex/registry/roster.json

Key concepts:
  ModelIdentity   — immutable descriptor for a single model variant
  ModelRoster     — the live set of known models (persisted to disk)
  FeatureSchema   — version-tagged feature contract with compatibility info
  FEATURE_SCHEMA_REGISTRY — canonical map of schema versions to FeatureSchema

Lifecycle statuses:
  active    — currently used for live inference
  candidate — being evaluated but not yet promoted
  retired   — no longer used for inference but kept for replay/comparison
  archived  — permanently shelved, not loaded at startup
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .feature_adapter import FEATURE_NAMES_V1, FEATURE_NAMES_V2_MACRO, FEATURE_NAMES


# ═══════════════════════════════════════════════════════════════════
# PATH HELPERS
# ═══════════════════════════════════════════════════════════════════

def _cortex_root() -> Path:
    ml_dir = Path(__file__).parent
    python_dir = ml_dir.parent
    aether_dir = python_dir.parent
    content_root = aether_dir.parent
    return content_root / "data_lake" / "cortex"


def _roster_path() -> Path:
    return _cortex_root() / "registry" / "roster.json"


# ═══════════════════════════════════════════════════════════════════
# FEATURE SCHEMA
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FeatureSchema:
    """
    Describes a versioned feature contract.

    Fields:
      version             — schema version string (e.g. "v1.0.0")
      feature_count       — expected dimensionality of the feature vector
      feature_names       — ordered list of feature names in the vector
      compatible_versions — other schema versions that can be adapted to this one
    """

    version: str
    feature_count: int
    feature_names: tuple[str, ...]
    compatible_versions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
            "compatible_versions": list(self.compatible_versions),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureSchema:
        return cls(
            version=d["version"],
            feature_count=d["feature_count"],
            feature_names=tuple(d.get("feature_names", [])),
            compatible_versions=tuple(d.get("compatible_versions", [])),
        )


# ── Canonical schema registry ──

FEATURE_SCHEMA_V1 = FeatureSchema(
    version="v1.0.0",
    feature_count=len(FEATURE_NAMES_V1),
    feature_names=tuple(FEATURE_NAMES_V1),
    compatible_versions=("v2.0.0",),  # v1 memories can be adapted into v2
)

FEATURE_SCHEMA_V2 = FeatureSchema(
    version="v2.0.0",
    feature_count=len(FEATURE_NAMES),
    feature_names=tuple(FEATURE_NAMES),
    compatible_versions=("v1.0.0",),  # v2 can read v1 memories via zero-padding
)

FEATURE_SCHEMA_REGISTRY: dict[str, FeatureSchema] = {
    "v1.0.0": FEATURE_SCHEMA_V1,
    "v2.0.0": FEATURE_SCHEMA_V2,
}


# ═══════════════════════════════════════════════════════════════════
# MODEL IDENTITY
# ═══════════════════════════════════════════════════════════════════

VALID_STATUSES = {"active", "candidate", "retired", "archived"}


@dataclass
class ModelIdentity:
    """
    Immutable descriptor for a single model variant.

    Fields:
      model_key              — unique key (e.g. "cortex_sgd_1h_24bar")
      feature_version        — which FeatureSchema this model expects
      label_policy_version   — which LabelPolicy was used for training labels
      description            — human-readable description
      created_utc            — ISO 8601 creation timestamp
      status                 — lifecycle status: active | candidate | retired | archived
    """

    model_key: str
    feature_version: str
    label_policy_version: str
    description: str = ""
    created_utc: str = ""
    status: str = "candidate"

    def __post_init__(self) -> None:
        if not self.created_utc:
            self.created_utc = datetime.now(timezone.utc).isoformat()
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid model status '{self.status}'. "
                f"Must be one of: {sorted(VALID_STATUSES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "feature_version": self.feature_version,
            "label_policy_version": self.label_policy_version,
            "description": self.description,
            "created_utc": self.created_utc,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelIdentity:
        return cls(
            model_key=d["model_key"],
            feature_version=d["feature_version"],
            label_policy_version=d["label_policy_version"],
            description=d.get("description", ""),
            created_utc=d.get("created_utc", ""),
            status=d.get("status", "candidate"),
        )


# ═══════════════════════════════════════════════════════════════════
# MODEL ROSTER
# ═══════════════════════════════════════════════════════════════════

class ModelRoster:
    """
    The live set of known models, persisted to roster.json.

    Supports registration, querying by status/feature-version/horizon,
    promotion, retirement, and schema compatibility checks.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelIdentity] = {}

    # ── Persistence ──

    def save(self) -> None:
        """Write the roster to disk (atomic via temp-rename)."""
        path = _roster_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "roster_version": "1.0",
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "models": {k: v.to_dict() for k, v in self._models.items()},
        }

        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            # Atomic replace (os.replace is atomic on POSIX, best-effort on Windows)
            os.replace(str(tmp), str(path))
            print(
                f"[ModelRegistry] Saved roster with {len(self._models)} model(s)",
                file=sys.stderr,
            )
        except Exception as ex:
            print(f"[ModelRegistry] Failed to save roster: {ex}", file=sys.stderr)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    @classmethod
    def load(cls) -> ModelRoster:
        """Load roster from disk, or return an empty roster seeded with the incumbent."""
        path = _roster_path()
        roster = cls()

        if not path.exists():
            print(
                "[ModelRegistry] No roster found, seeding with incumbent",
                file=sys.stderr,
            )
            roster._seed_incumbent()
            roster.save()
            return roster

        try:
            with open(path, "r") as f:
                data = json.load(f)

            models_raw = data.get("models", {})
            for key, model_dict in models_raw.items():
                try:
                    roster._models[key] = ModelIdentity.from_dict(model_dict)
                except Exception as ex:
                    print(
                        f"[ModelRegistry] Skipping corrupt entry '{key}': {ex}",
                        file=sys.stderr,
                    )

            print(
                f"[ModelRegistry] Loaded roster with {len(roster._models)} model(s)",
                file=sys.stderr,
            )
        except Exception as ex:
            print(f"[ModelRegistry] Failed to load roster: {ex}", file=sys.stderr)
            roster._seed_incumbent()

        return roster

    def _seed_incumbent(self) -> None:
        """Register the well-known incumbent model."""
        incumbent = ModelIdentity(
            model_key="cortex_sgd_1h_24bar",
            feature_version="v2.0.0",
            label_policy_version="lp_v1",
            description="Incumbent SGD classifier, 1h interval, 24-bar horizon, v2 features",
            status="active",
        )
        self._models[incumbent.model_key] = incumbent

    # ── Registration ──

    def register_model(self, identity: ModelIdentity, *, overwrite: bool = False) -> bool:
        """
        Register a new model identity.

        Args:
          identity  — the ModelIdentity to register
          overwrite — if True, replace an existing entry with the same key

        Returns:
          True if registered, False if the key already exists and overwrite is False.
        """
        if identity.model_key in self._models and not overwrite:
            print(
                f"[ModelRegistry] Model '{identity.model_key}' already exists "
                f"(use overwrite=True to replace)",
                file=sys.stderr,
            )
            return False

        self._models[identity.model_key] = identity
        print(
            f"[ModelRegistry] Registered '{identity.model_key}' "
            f"(feature={identity.feature_version}, status={identity.status})",
            file=sys.stderr,
        )
        return True

    # ── Queries ──

    def get_model(self, model_key: str) -> ModelIdentity | None:
        """Return a model identity by key, or None if not found."""
        return self._models.get(model_key)

    def list_models(
        self,
        *,
        status: str | None = None,
        feature_version: str | None = None,
    ) -> list[ModelIdentity]:
        """
        List models, optionally filtered by status and/or feature version.

        Args:
          status          — filter to this lifecycle status (e.g. "active")
          feature_version — filter to models expecting this feature schema
        """
        results: list[ModelIdentity] = []
        for model in self._models.values():
            if status is not None and model.status != status:
                continue
            if feature_version is not None and model.feature_version != feature_version:
                continue
            results.append(model)
        return results

    def get_active_models(self) -> list[ModelIdentity]:
        """Return all models with status 'active'."""
        return self.list_models(status="active")

    def get_active_model_for_horizon(self, horizon: str) -> ModelIdentity | None:
        """
        Find the active model whose key contains the given horizon tag.

        This is a convention-based lookup: model keys embed their horizon
        (e.g. "cortex_sgd_1h_24bar" contains "1h").

        Returns the first match, or None.
        """
        active = self.get_active_models()
        for model in active:
            if horizon in model.model_key:
                return model
        # Fallback: return any active model if only one exists
        if len(active) == 1:
            return active[0]
        return None

    def get_models_compatible_with(self, feature_version: str) -> list[ModelIdentity]:
        """
        Return models whose feature_version matches directly OR whose schema
        is listed as compatible via the FEATURE_SCHEMA_REGISTRY.
        """
        compatible_versions = get_compatible_schemas(feature_version)
        results: list[ModelIdentity] = []
        for model in self._models.values():
            if model.feature_version in compatible_versions:
                results.append(model)
        return results

    # ── Lifecycle transitions ──

    def promote_model(self, model_key: str) -> bool:
        """
        Promote a candidate model to active status.

        Any currently active models with the same feature_version are retired
        to make room for the new incumbent.

        Returns True if promotion succeeded, False if the model was not found
        or was not in 'candidate' status.
        """
        model = self._models.get(model_key)
        if model is None:
            print(
                f"[ModelRegistry] Cannot promote '{model_key}': not found",
                file=sys.stderr,
            )
            return False

        if model.status != "candidate":
            print(
                f"[ModelRegistry] Cannot promote '{model_key}': "
                f"status is '{model.status}', expected 'candidate'",
                file=sys.stderr,
            )
            return False

        # Retire the current active model(s) for this feature version
        for other in self._models.values():
            if (
                other.model_key != model_key
                and other.status == "active"
                and other.feature_version == model.feature_version
            ):
                other.status = "retired"
                print(
                    f"[ModelRegistry] Retired '{other.model_key}' "
                    f"(displaced by '{model_key}')",
                    file=sys.stderr,
                )

        model.status = "active"
        print(
            f"[ModelRegistry] Promoted '{model_key}' to active",
            file=sys.stderr,
        )
        return True

    def retire_model(self, model_key: str) -> bool:
        """
        Retire a model (set status to 'retired').

        Returns True if retirement succeeded, False if not found.
        """
        model = self._models.get(model_key)
        if model is None:
            print(
                f"[ModelRegistry] Cannot retire '{model_key}': not found",
                file=sys.stderr,
            )
            return False

        prev_status = model.status
        model.status = "retired"
        print(
            f"[ModelRegistry] Retired '{model_key}' ({prev_status} -> retired)",
            file=sys.stderr,
        )
        return True

    # ── Serialization ──

    def to_dict(self) -> dict[str, Any]:
        """Return the full roster as a serializable dict."""
        return {
            "roster_version": "1.0",
            "model_count": len(self._models),
            "models": {k: v.to_dict() for k, v in self._models.items()},
        }


# ═══════════════════════════════════════════════════════════════════
# SCHEMA COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════

def check_schema_compatibility(source_version: str, target_version: str) -> dict[str, Any]:
    """
    Check whether memories with source_version features can be used
    by a model expecting target_version features.

    Returns a dict with:
      compatible       — True if the memory can be consumed (directly or via adapter)
      reason           — human-readable explanation
      adapter_available — True if an adapter exists to bridge the gap
      source_version   — echo of input
      target_version   — echo of input
    """
    if source_version == target_version:
        return {
            "compatible": True,
            "reason": "exact_match",
            "adapter_available": False,
            "source_version": source_version,
            "target_version": target_version,
        }

    source_schema = FEATURE_SCHEMA_REGISTRY.get(source_version)
    target_schema = FEATURE_SCHEMA_REGISTRY.get(target_version)

    if source_schema is None:
        return {
            "compatible": False,
            "reason": f"unknown_source_schema:{source_version}",
            "adapter_available": False,
            "source_version": source_version,
            "target_version": target_version,
        }

    if target_schema is None:
        return {
            "compatible": False,
            "reason": f"unknown_target_schema:{target_version}",
            "adapter_available": False,
            "source_version": source_version,
            "target_version": target_version,
        }

    # Check if target lists source as compatible (adapter path exists)
    if source_version in target_schema.compatible_versions:
        return {
            "compatible": True,
            "reason": f"adapter_available:{source_version}->{target_version}",
            "adapter_available": True,
            "source_version": source_version,
            "target_version": target_version,
        }

    # Check reverse compatibility (source lists target)
    if target_version in source_schema.compatible_versions:
        return {
            "compatible": True,
            "reason": f"adapter_available:{source_version}->{target_version}",
            "adapter_available": True,
            "source_version": source_version,
            "target_version": target_version,
        }

    return {
        "compatible": False,
        "reason": f"incompatible_schemas:{source_version}<-/->{target_version}",
        "adapter_available": False,
        "source_version": source_version,
        "target_version": target_version,
    }


def get_compatible_schemas(version: str) -> set[str]:
    """
    Return the set of schema versions compatible with the given version.

    Always includes the version itself. Adds any versions listed in the
    schema's compatible_versions field.
    """
    result = {version}
    schema = FEATURE_SCHEMA_REGISTRY.get(version)
    if schema is not None:
        result.update(schema.compatible_versions)
    return result
