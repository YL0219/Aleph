using System.Text.Json;

namespace Aleph;

/// <summary>
/// Universal blood cell — the base type for all internal circulatory events.
/// Immutable record. Safe to pass across async boundaries.
/// </summary>
public abstract record AlephEvent
{
    public Guid EventId { get; init; } = Guid.NewGuid();
    public required DateTimeOffset OccurredAtUtc { get; init; }
    public required string Source { get; init; }
    public required string Kind { get; init; }
    public required PulseSeverity Severity { get; init; }
    public IReadOnlyList<string>? Tags { get; init; }
    public Guid? CorrelationId { get; init; }
    public Guid? CausationId { get; init; }
}

/// <summary>
/// Blood cell representing an external or internal stress change in the autonomic system.
/// Published by Homeostasis when it ingests an ExternalStress PulseEnvelope.
/// </summary>
public sealed record AutonomicStressEvent : AlephEvent
{
    public required double StressLevel { get; init; }
    public required double FatigueLevel { get; init; }
    public required double OverloadLevel { get; init; }
    public required int FailureStreak { get; init; }
    public string? Message { get; init; }
    public IReadOnlyDictionary<string, double>? Metrics { get; init; }
}

/// <summary>
/// Blood cell representing the result of a single heartbeat pulse.
/// Published by Homeostasis when it records a pulse report.
/// </summary>
public sealed record HeartbeatPulseEvent : AlephEvent
{
    public required long DurationMs { get; init; }
    public required bool Success { get; init; }
    public required double StressLevel { get; init; }
    public required double FatigueLevel { get; init; }
    public required double OverloadLevel { get; init; }
    public required int FailureStreak { get; init; }
    public string? Message { get; init; }
    public IReadOnlyDictionary<string, double>? Metrics { get; init; }
}

/// <summary>
/// Blood cell representing a market data perception event.
/// Published by the Axiom market gateway when live market data enters the bloodstream.
/// </summary>
public sealed record MarketDataEvent : AlephEvent
{
    public required string Symbol { get; init; }
    public required string Interval { get; init; }
    public required bool Success { get; init; }

    /// <summary>How the data was sourced: live_bootstrap, live_refresh, live_quote_overlay.</summary>
    public required string SourceKind { get; init; }

    public int? RowsWritten { get; init; }
    public string? ParquetPath { get; init; }
    public string? ErrorMessage { get; init; }
    public double? LatestPrice { get; init; }
    public string? QuoteTimestampUtc { get; init; }
}

/// <summary>
/// Blood cell representing a perception refresh — macro proxies, calendar, headlines.
/// Published by the MarketIngestionOrchestrator after running perception ingest.
/// </summary>
public sealed record PerceptionRefreshEvent : AlephEvent
{
    public required bool Success { get; init; }
    public required long DurationMs { get; init; }

    // Macro proxies
    public int ProxiesSucceeded { get; init; }
    public int ProxiesTotal { get; init; }

    // Calendar
    public bool CalendarOk { get; init; }
    public string? CalendarProvider { get; init; }
    public int CalendarEventCount { get; init; }

    // Headlines
    public bool HeadlinesOk { get; init; }
    public string? HeadlinesProvider { get; init; }
    public int HeadlineCount { get; init; }

    public string? ManifestPath { get; init; }
    public string? ErrorMessage { get; init; }
}

/// <summary>
/// Blood cell summarizing one completed Sleep Cycle (resolve → train).
/// Published by SleepCycleService at the end of each cycle for observability.
/// v2: adds cycle numbering, scorecard fields, stall detection, evaluation results.
/// </summary>
public sealed record SleepCycleSummaryEvent : AlephEvent
{
    // Identity
    public required string Symbol { get; init; }
    public required string Horizon { get; init; }
    public required long DurationMs { get; init; }
    public int CycleNumber { get; init; }

    // Phase results
    public bool StatusOk { get; init; }
    public bool ResolveOk { get; init; }
    public bool TrainOk { get; init; }
    public string? SkipReason { get; init; }

    // Pending / resolved counts
    public int PendingCount { get; init; }
    public int PendingEligible { get; init; }
    public int PendingBlocked { get; init; }
    public int NewlyResolved { get; init; }
    public int Deferred { get; init; }
    public int Expired { get; init; }
    public int Errored { get; init; }

    // Training
    public int SamplesFitted { get; init; }
    public int FreshCount { get; init; }
    public int ReplayCount { get; init; }
    public string? TrainingGate { get; init; }
    public string? TrainingGateReason { get; init; }
    public int CursorSequence { get; init; }

    // Model
    public string? ModelState { get; init; }
    public int TrainedSamples { get; init; }

    // Quality metrics
    public double ResolveAccuracy { get; init; }
    public double ResolveMeanBrier { get; init; }
    public string? ClassSkewWarning { get; init; }
    public IReadOnlyList<string>? DriftFlags { get; init; }
    public IReadOnlyDictionary<string, int>? ResolvedClassDistribution { get; init; }
    public IReadOnlyDictionary<string, int>? TrainClassDistribution { get; init; }

    // Homeostasis at cycle time
    public double StressLevel { get; init; }
    public double FatigueLevel { get; init; }

    // Cycle scorecard (from resolve — this batch's quality)
    public double CycleScorecardAccuracy { get; init; }
    public double CycleScorecardBrier { get; init; }
    public string? CycleScorecardGrade { get; init; }

    // Rolling scorecard (from status — overall model quality)
    public double RollingScorecardAccuracy { get; init; }
    public double RollingScorecardBrier { get; init; }
    public string? RollingScorecardGrade { get; init; }
    public int RollingScorecardSamples { get; init; }

    // Stall detection
    public int ConsecutiveZeroProgressCycles { get; init; }
    public double HoursSinceLastProgress { get; init; }

    // Periodic evaluation results
    public bool EvaluationRan { get; init; }
    public bool EvaluationOk { get; init; }
    public int EvaluationChallengerCount { get; init; }
    public string? EvaluationBestChallenger { get; init; }
    public string? EvaluationPromotionDecision { get; init; }
}
