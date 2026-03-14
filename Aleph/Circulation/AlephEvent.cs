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
