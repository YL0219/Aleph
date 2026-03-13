namespace Aleph;

/// <summary>
/// Bounded in-memory record of an autonomic state change.
/// Used for short-term observability within Homeostasis — NOT persistent storage.
/// Renamed from AutonomicEvent to avoid collision with the EF Core entity.
/// </summary>
public sealed record AutonomicEventRecord
{
    public required DateTimeOffset TimestampUtc { get; init; }
    public required string Source { get; init; }
    public required string Kind { get; init; }
    public required PulseSeverity Severity { get; init; }
    public required string Message { get; init; }
    public IReadOnlyDictionary<string, double>? Metrics { get; init; }
    public IReadOnlyList<string>? Tags { get; init; }
}
