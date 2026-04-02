// CONTRACT / INVARIANTS
// - Singleton service that receives rejected blood cells from the BloodFilter immune system.
// - Maintains a bounded in-memory ring buffer of quarantine records for diagnostics.
// - Logs every quarantine event at Warning level for observability.
// - Thread-safe: all mutations go through lock; all reads return immutable snapshots.
// - Does NOT persist quarantine records to disk — that's the Kidneys' job if needed.
// - Exposes metrics (total count, recent anomalies) for Homeostasis/Heartbeat monitoring.

namespace Aleph;

/// <summary>
/// A single quarantine record — one rejected blood cell with its anomaly diagnosis.
/// </summary>
public sealed record QuarantineRecord
{
    public required DateTimeOffset QuarantinedAtUtc { get; init; }
    public required string OrganName { get; init; }
    public required string EventKind { get; init; }
    public required string Anomaly { get; init; }
    public Guid? EventId { get; init; }
    public string? Symbol { get; init; }
}

/// <summary>
/// The Quarantine Ward — isolation for corrupted or malformed blood cells
/// rejected by the BloodFilter immune system.
///
/// Organs call Quarantine.Isolate() when a screening fails. The record is
/// logged and buffered for diagnostic queries. This prevents poisoned data
/// from reaching ML models or trading logic.
/// </summary>
public sealed class Quarantine
{
    private const int MaxRecords = 500;

    private readonly object _lock = new();
    private readonly LinkedList<QuarantineRecord> _records = new();
    private readonly ILogger<Quarantine> _logger;
    private long _totalQuarantined;
    private long _totalEvicted;

    public Quarantine(ILogger<Quarantine> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Total number of blood cells quarantined since application start.
    /// </summary>
    public long TotalQuarantined
    {
        get { lock (_lock) { return _totalQuarantined; } }
    }

    /// <summary>
    /// Total records evicted from the ring buffer due to overflow.
    /// When this grows, evidence is being destroyed. Monitor this.
    /// </summary>
    public long TotalEvicted
    {
        get { lock (_lock) { return _totalEvicted; } }
    }

    /// <summary>
    /// Isolate a rejected blood cell. Logs the anomaly and stores the record.
    /// </summary>
    public void Isolate(string organName, AlephEvent? evt, BloodScreening screening)
    {
        if (screening.IsHealthy) return; // Nothing to quarantine

        var record = new QuarantineRecord
        {
            QuarantinedAtUtc = DateTimeOffset.UtcNow,
            OrganName = organName,
            EventKind = evt?.Kind ?? "unknown",
            Anomaly = screening.Anomaly ?? "unspecified",
            EventId = evt?.EventId,
            Symbol = ExtractSymbol(evt),
        };

        lock (_lock)
        {
            _records.AddFirst(record);
            _totalQuarantined++;

            while (_records.Count > MaxRecords)
            {
                _records.RemoveLast();
                _totalEvicted++;
            }
        }

        _logger.LogWarning(
            "[Quarantine] {Organ} rejected {Kind} ({EventId}): {Anomaly}",
            record.OrganName, record.EventKind, record.EventId, record.Anomaly);
    }

    /// <summary>
    /// Isolate a rejected Python output (no AlephEvent available yet).
    /// </summary>
    public void IsolatePythonOutput(string organName, string action, string anomaly, string? symbol = null)
    {
        var record = new QuarantineRecord
        {
            QuarantinedAtUtc = DateTimeOffset.UtcNow,
            OrganName = organName,
            EventKind = $"python_output/{action}",
            Anomaly = anomaly,
            Symbol = symbol,
        };

        lock (_lock)
        {
            _records.AddFirst(record);
            _totalQuarantined++;

            while (_records.Count > MaxRecords)
            {
                _records.RemoveLast();
                _totalEvicted++;
            }
        }

        _logger.LogWarning(
            "[Quarantine] {Organ} rejected Python output for {Action}: {Anomaly}",
            organName, action, anomaly);
    }

    /// <summary>
    /// Returns the most recent quarantine records, newest first.
    /// </summary>
    public IReadOnlyList<QuarantineRecord> GetRecent(int maxCount = 50)
    {
        lock (_lock)
        {
            return _records.Take(Math.Min(maxCount, _records.Count)).ToList().AsReadOnly();
        }
    }

    /// <summary>
    /// Returns a complete diagnostic snapshot for the DiagnosticsController.
    /// Includes counters, overflow status, and recent records.
    /// </summary>
    public QuarantineDiagnosticSnapshot GetDiagnosticSnapshot(int maxRecords = 100)
    {
        lock (_lock)
        {
            return new QuarantineDiagnosticSnapshot
            {
                TotalQuarantined = _totalQuarantined,
                TotalEvicted = _totalEvicted,
                BufferCapacity = MaxRecords,
                CurrentBufferSize = _records.Count,
                EvidenceLoss = _totalEvicted > 0,
                RecentRecords = _records.Take(Math.Min(maxRecords, _records.Count)).ToList().AsReadOnly(),
                SnapshotAtUtc = DateTimeOffset.UtcNow,
            };
        }
    }

    private static string? ExtractSymbol(AlephEvent? evt) => evt switch
    {
        MarketDataEvent mde => mde.Symbol,
        MetabolicEvent me => me.Symbol,
        PredictionEvent pe => pe.Symbol,
        _ => null,
    };
}

/// <summary>
/// Immutable snapshot of the Quarantine ward's state for diagnostic exposure.
/// </summary>
public sealed record QuarantineDiagnosticSnapshot
{
    public required long TotalQuarantined { get; init; }
    public required long TotalEvicted { get; init; }
    public required int BufferCapacity { get; init; }
    public required int CurrentBufferSize { get; init; }

    /// <summary>True if any records have been evicted from the ring buffer. Evidence loss.</summary>
    public required bool EvidenceLoss { get; init; }

    public required IReadOnlyList<QuarantineRecord> RecentRecords { get; init; }
    public required DateTimeOffset SnapshotAtUtc { get; init; }
}
