namespace Aleph;

/// <summary>
/// Refined blood cell produced by the ML Cortex after processing a MetabolicEvent.
/// Carries a probabilistic prediction for downstream consumers (primarily Arbiter).
///
/// This is a prediction, NOT a trade decision. Arbiter decides what to do with it.
///
/// Key properties:
///   - 3-class probabilities (bullish / neutral / bearish)
///   - Regime and event probabilities for macro-aware reasoning
///   - Temporal security enforcement (anti-leakage)
///   - Confidence and action tendency for Arbiter reasoning
///   - Model state tracking (cold_start → warming → active)
///   - Provenance back to the source MetabolicEvent
///   - Multi-horizon-ready via ActiveHorizon key
/// </summary>
public sealed record PredictionEvent : AlephEvent
{
    // ─── Identity ────────────────────────────────────────────────────

    /// <summary>Unique prediction identifier for tracing through pending/resolved pipeline.</summary>
    public required string PredictionId { get; init; }

    public required string Symbol { get; init; }
    public required string Interval { get; init; }
    public required string ActiveHorizon { get; init; }
    public required string AsOfUtc { get; init; }

    // ─── Provenance ──────────────────────────────────────────────────

    /// <summary>EventId of the MetabolicEvent that triggered this prediction.</summary>
    public required Guid SourceMetabolicEventId { get; init; }

    /// <summary>Version identifier of the model that produced this prediction.</summary>
    public required string ModelVersion { get; init; }

    /// <summary>Compound key identifying the model (e.g. "cortex_sgd_1h_24bar").</summary>
    public required string ModelKey { get; init; }

    /// <summary>Version of the feature extraction contract (e.g. "v2.0.0").</summary>
    public required string FeatureVersion { get; init; }

    // ─── Temporal Security ───────────────────────────────────────────

    /// <summary>Observation cutoff from the source MetabolicEvent temporal envelope.</summary>
    public required string SourceObservationCutoffUtc { get; init; }

    /// <summary>Whether the temporal security check passed (no look-ahead detected).</summary>
    public required bool TemporalSecurityPassed { get; init; }

    /// <summary>
    /// Whether this prediction's pending sample is eligible for training.
    /// Must be false if TemporalSecurityPassed is false.
    /// </summary>
    public required bool EligibleForTraining { get; init; }

    // ─── Model State ─────────────────────────────────────────────────

    /// <summary>Current state of the model: cold_start, warming, active.</summary>
    public required string ModelState { get; init; }

    /// <summary>Number of samples the model has been trained on.</summary>
    public int TrainedSamples { get; init; }

    // ─── Prediction Output ───────────────────────────────────────────

    /// <summary>Predicted class: bullish, neutral, or bearish.</summary>
    public required string PredictedClass { get; init; }

    /// <summary>Class probabilities (bullish, neutral, bearish).</summary>
    public required PredictionProbabilities Probabilities { get; init; }

    /// <summary>Overall confidence in the prediction [0, 1].</summary>
    public required double Confidence { get; init; }

    /// <summary>
    /// Directional tendency in [-1, 1].
    /// Negative = bearish lean, 0 = neutral, Positive = bullish lean.
    /// Arbiter uses this as a continuous signal, not a binary decision.
    /// </summary>
    public required double ActionTendency { get; init; }

    // ─── Macro-Aware Probabilities ───────────────────────────────────

    /// <summary>Regime probability distribution for macro-aware reasoning.</summary>
    public PredictionRegimeProbabilities? RegimeProbabilities { get; init; }

    /// <summary>Event-driven probability signals.</summary>
    public PredictionEventProbabilities? EventProbabilities { get; init; }

    /// <summary>Priority score [0,1] — how actionable this prediction is for Arbiter.</summary>
    public double? PriorityScore { get; init; }

    // ─── Drivers & Risks ─────────────────────────────────────────────

    /// <summary>Top factors driving this prediction.</summary>
    public IReadOnlyList<string> TopDrivers { get; init; } = Array.Empty<string>();

    /// <summary>Top risk factors that could invalidate this prediction.</summary>
    public IReadOnlyList<string> TopRisks { get; init; } = Array.Empty<string>();

    /// <summary>Upcoming catalysts being watched by the model.</summary>
    public IReadOnlyList<PredictionCatalystRef> WatchedCatalysts { get; init; } = Array.Empty<PredictionCatalystRef>();

    // ─── Training Status (lightweight) ───────────────────────────────

    /// <summary>Whether a pending sample was stored for future training.</summary>
    public bool PendingSampleStored { get; init; }

    /// <summary>Whether an incremental training update occurred on this cycle.</summary>
    public bool TrainingOccurred { get; init; }

    // ─── Warnings ────────────────────────────────────────────────────

    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();
}

// ═════════════════════════════════════════════════════════════════════
// Existing nested records (unchanged)
// ═════════════════════════════════════════════════════════════════════

/// <summary>
/// 3-class probability distribution for market direction prediction.
/// Values should sum to ~1.0.
/// </summary>
public sealed record PredictionProbabilities
{
    public required double Bullish { get; init; }
    public required double Neutral { get; init; }
    public required double Bearish { get; init; }
}

// ═════════════════════════════════════════════════════════════════════
// NEW: Macro-aware probability records
// ═════════════════════════════════════════════════════════════════════

/// <summary>
/// Regime probability distribution for macro-aware reasoning by Arbiter.
/// Values are independent probabilities [0,1], not required to sum to 1.
/// </summary>
public sealed record PredictionRegimeProbabilities
{
    public double RiskOn { get; init; }
    public double RiskOff { get; init; }
    public double InflationPressure { get; init; }
    public double GrowthScare { get; init; }
    public double PolicyShock { get; init; }
    public double FlightToSafety { get; init; }
}

/// <summary>
/// Event-driven probability signals for macro-aware reasoning.
/// </summary>
public sealed record PredictionEventProbabilities
{
    public double Materiality { get; init; }
    public double FollowThrough { get; init; }
    public double VolatilityExpansion { get; init; }
}

/// <summary>
/// Reference to an upcoming catalyst that the model is aware of.
/// </summary>
public sealed record PredictionCatalystRef
{
    public required string EventType { get; init; }
    public required string ScheduledForUtc { get; init; }
    public required double ImportanceProbability { get; init; }
}
