namespace Aleph;

/// <summary>
/// Refined blood cell produced by the Liver after digesting raw MarketDataEvent candles.
/// Downstream organs (ML, Trading) should consume MetabolicEvents rather than
/// re-digesting raw candle data themselves.
///
/// Carries a canonical feature snapshot, factor scores, and provenance metadata
/// so consumers never need to re-run the indicator pipeline.
/// </summary>
public sealed record MetabolicEvent : AlephEvent
{
    // ─── Identity ────────────────────────────────────────────────────

    public required string Symbol { get; init; }
    public required string Interval { get; init; }
    public required string AsOfUtc { get; init; }

    // ─── Provenance ──────────────────────────────────────────────────

    /// <summary>EventId of the MarketDataEvent that triggered this digestion.</summary>
    public required Guid SourceEventId { get; init; }

    /// <summary>Path to the source parquet file that was digested.</summary>
    public string? SourceParquetPath { get; init; }

    /// <summary>Schema version of the metabolic digestion contract.</summary>
    public required string MetabolicVersion { get; init; }

    // ─── Data Quality ────────────────────────────────────────────────

    public required int RowCount { get; init; }
    public required bool EnoughForLongTrend { get; init; }
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();

    // ─── Feature Snapshot (canonical baseline indicators) ────────────

    public required MetabolicSnapshot Snapshot { get; init; }

    // ─── Factor Scores (canonical scoring output) ────────────────────

    public required MetabolicFactorScores FactorScores { get; init; }

    // ─── Composite Probability ───────────────────────────────────────

    public required MetabolicComposite Composite { get; init; }

    // ─── Conclusion ──────────────────────────────────────────────────

    public required string Bias { get; init; }
    public required double Confidence { get; init; }
    public IReadOnlyList<string> KeyDrivers { get; init; } = Array.Empty<string>();
    public IReadOnlyList<string> Risks { get; init; } = Array.Empty<string>();

    // ─── Recent Windows (small processed tails for downstream) ───────

    public MetabolicRecentWindows? RecentWindows { get; init; }

    // ─── Persistence ─────────────────────────────────────────────────

    /// <summary>Path where the metabolized artifact was persisted.</summary>
    public string? MetabolizedArtifactPath { get; init; }

    // ─── Temporal Envelope (anti-leakage / point-in-time safety) ─────

    /// <summary>Temporal provenance proving point-in-time safety for training eligibility.</summary>
    public MetabolicTemporalEnvelope? Temporal { get; init; }

    // ─── Macro Context (cross-asset, scheduled, headline, crypto) ────

    /// <summary>Macro-aware context enrichment. Nullable because not all events carry macro data.</summary>
    public MetabolicMacroContext? MacroContext { get; init; }

    // ─── Context Coverage (observability of what was/wasn't available) ─

    public MetabolicContextCoverage? ContextCoverage { get; init; }
}

// ═════════════════════════════════════════════════════════════════════
// Existing nested records (unchanged)
// ═════════════════════════════════════════════════════════════════════

/// <summary>
/// Canonical baseline technical indicators at a single point in time.
/// Mirrors the output of Aether's quant/analysis.py snapshot.
/// </summary>
public sealed record MetabolicSnapshot
{
    public double? Price { get; init; }
    public double? Sma20 { get; init; }
    public double? Sma50 { get; init; }
    public double? Sma200 { get; init; }
    public double? Ema12 { get; init; }
    public double? Ema26 { get; init; }
    public double? Rsi14 { get; init; }
    public MetabolicMacd? Macd { get; init; }
    public MetabolicBollinger? Bollinger { get; init; }
    public double? Atr14 { get; init; }
    public double? AtrPct { get; init; }
    public double? Volatility20 { get; init; }
    public double? VolumeSma20 { get; init; }

    /// <summary>Price distance from key moving averages (percentage).</summary>
    public double? DistSma20 { get; init; }
    public double? DistSma50 { get; init; }
    public double? DistSma200 { get; init; }
}

public sealed record MetabolicMacd
{
    public double? Line { get; init; }
    public double? Signal { get; init; }
    public double? Histogram { get; init; }
}

public sealed record MetabolicBollinger
{
    public double? Mid { get; init; }
    public double? Upper { get; init; }
    public double? Lower { get; init; }
    public double? Bandwidth { get; init; }
}

/// <summary>
/// Canonical factor scores from Aether's scoring pipeline.
/// Each factor has a score in [-1, 1] and a descriptive label.
/// </summary>
public sealed record MetabolicFactorScores
{
    public required MetabolicFactor Trend { get; init; }
    public required MetabolicFactor Momentum { get; init; }
    public required MetabolicFactor Volatility { get; init; }
    public required MetabolicFactor Participation { get; init; }
}

public sealed record MetabolicFactor
{
    public required double Score { get; init; }
    public required string Label { get; init; }
    public string? Reason { get; init; }
}

/// <summary>
/// Composite probability output from Aether's scoring pipeline.
/// </summary>
public sealed record MetabolicComposite
{
    public required double BullishProbability { get; init; }
    public required double BearishProbability { get; init; }
    public required double NeutralProbability { get; init; }
    public required double Confidence { get; init; }
}

/// <summary>
/// Small recent processed windows — the last N values of key series.
/// Allows downstream consumers to see recent trajectory without loading full history.
/// </summary>
public sealed record MetabolicRecentWindows
{
    public IReadOnlyList<double?>? Close { get; init; }
    public IReadOnlyList<double?>? Rsi14 { get; init; }
    public IReadOnlyList<double?>? MacdHistogram { get; init; }
    public IReadOnlyList<double?>? AtrPct { get; init; }
}

// ═════════════════════════════════════════════════════════════════════
// NEW: Temporal Envelope — anti-leakage provenance
// ═════════════════════════════════════════════════════════════════════

/// <summary>
/// Temporal provenance envelope proving point-in-time safety.
/// A sample is only eligible for training if PointInTimeSafe == true.
/// Rule: every included context item must have KnowledgeUtc &lt;= ObservationCutoffUtc.
/// </summary>
public sealed record MetabolicTemporalEnvelope
{
    public required string BarOpenUtc { get; init; }
    public required string BarCloseUtc { get; init; }
    public required string ObservationCutoffUtc { get; init; }
    public required string FeatureBuiltUtc { get; init; }
    public required string MaxIncludedKnowledgeUtc { get; init; }
    public required string TemporalPolicyVersion { get; init; }
    public required bool PointInTimeSafe { get; init; }
    public bool HistoricalReplayMode { get; init; }
    public IReadOnlyList<string> ExclusionReasons { get; init; } = Array.Empty<string>();
}

// ═════════════════════════════════════════════════════════════════════
// NEW: Macro Context — cross-asset, scheduled, headline, crypto
// ═════════════════════════════════════════════════════════════════════

/// <summary>
/// Macro-aware context enrichment carried as compact scores/tags only.
/// No raw news text — only refs, tags, and pre-computed scores.
/// </summary>
public sealed record MetabolicMacroContext
{
    public MetabolicCrossAssetSnapshot? CrossAsset { get; init; }
    public MetabolicScheduledContext? Scheduled { get; init; }
    public MetabolicHeadlineContext? Headlines { get; init; }
    public MetabolicCryptoStressContext? CryptoStress { get; init; }
    public MetabolicRegimeHints? RegimeHints { get; init; }
    public IReadOnlyList<string> MacroTags { get; init; } = Array.Empty<string>();
}

public sealed record MetabolicCrossAssetSnapshot
{
    public required string AsOfUtc { get; init; }
    public double? EquitiesRiskScore { get; init; }
    public double? BondsRiskScore { get; init; }
    public double? GoldStrengthScore { get; init; }
    public double? SilverStrengthScore { get; init; }
    public double? DollarPressureScore { get; init; }
    public double? VolatilityPressureScore { get; init; }
    public double? CryptoRiskScore { get; init; }
    public double? LiquidityStressScore { get; init; }
    public double? CorrelationStressScore { get; init; }
    public double? CoverageScore { get; init; }
}

public sealed record MetabolicScheduledContext
{
    public IReadOnlyList<MetabolicKnownCatalyst> UpcomingCatalysts { get; init; } = Array.Empty<MetabolicKnownCatalyst>();
    public bool HighPriorityEventWithin24h { get; init; }
    public double? ScheduleTensionScore { get; init; }
    public double? CalendarCoverageScore { get; init; }
}

public sealed record MetabolicKnownCatalyst
{
    public required string CatalystId { get; init; }
    public required string EventType { get; init; }
    public required string ScheduledForUtc { get; init; }
    public required string KnowledgeUtc { get; init; }
    public required double PriorityProbability { get; init; }
    public IReadOnlyList<string> ExpectedAffectedAssets { get; init; } = Array.Empty<string>();
}

public sealed record MetabolicHeadlineContext
{
    public IReadOnlyList<string> ActiveTags { get; init; } = Array.Empty<string>();
    public IReadOnlyList<string> HeadlineRefs { get; init; } = Array.Empty<string>();
    public int HeadlineCount { get; init; }
    public double? MaterialityScore { get; init; }
    public double? ShockScore { get; init; }
    public double? SourceDiversityScore { get; init; }
    public string? MaxIncludedKnowledgeUtc { get; init; }
}

public sealed record MetabolicCryptoStressContext
{
    public required string AsOfUtc { get; init; }
    public double? CryptoRiskScore { get; init; }
    public double? CryptoVolatilityScore { get; init; }
    public double? WeekendStressScore { get; init; }
    public double? StablecoinStressScore { get; init; }
}

public sealed record MetabolicRegimeHints
{
    public double? RiskOnProbability { get; init; }
    public double? RiskOffProbability { get; init; }
    public double? InflationPressureProbability { get; init; }
    public double? GrowthScareProbability { get; init; }
    public double? PolicyShockProbability { get; init; }
    public double? FlightToSafetyProbability { get; init; }
    public double? RegimeConfidence { get; init; }
}

// ═════════════════════════════════════════════════════════════════════
// NEW: Context Coverage — observability of what was/wasn't available
// ═════════════════════════════════════════════════════════════════════

public sealed record MetabolicContextCoverage
{
    public bool HadHeadlineContext { get; init; }
    public bool HadScheduledContext { get; init; }
    public bool HadCryptoContext { get; init; }
    public IReadOnlyList<string> MissingContextReasons { get; init; } = Array.Empty<string>();
}
