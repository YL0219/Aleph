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
}

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
