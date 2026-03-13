using System.ComponentModel.DataAnnotations;

namespace Aleph;

/// <summary>
/// EF Core entity for persisted autonomic and heartbeat events (the Kidneys).
/// Written by AutonomicPersistenceService from the AlephBus.
/// </summary>
public class AutonomicEvent
{
    [Key]
    public int Id { get; set; }

    public DateTime OccurredAtUtc { get; set; }

    [MaxLength(128)]
    public string Source { get; set; } = string.Empty;

    [MaxLength(64)]
    public string Kind { get; set; } = string.Empty;

    public int Severity { get; set; }

    [MaxLength(1024)]
    public string? Message { get; set; }

    /// <summary>JSON-serialized metrics dictionary for later analysis.</summary>
    public string? MetricsJson { get; set; }

    /// <summary>JSON-serialized tags list.</summary>
    public string? TagsJson { get; set; }

    public double StressLevel { get; set; }
    public double FatigueLevel { get; set; }
    public double OverloadLevel { get; set; }
    public int FailureStreak { get; set; }

    /// <summary>Optional correlation ID for tracing related events.</summary>
    public Guid? CorrelationId { get; set; }
}
