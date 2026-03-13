namespace Aleph;

/// <summary>
/// Thread-safe singleton implementing autonomic regulation.
/// Tracks external stress (market danger) and internal fatigue (system overload).
/// All mutations go through lock; all reads return immutable snapshots.
/// Implements IStressInjector as the narrow write-only pathway for external domains.
/// </summary>
public sealed class Homeostasis : IHomeostasis, IStressInjector
{
    private readonly object _lock = new();
    private readonly ILogger<Homeostasis> _logger;
    private readonly IAlephBus _bus;

    // ── Mutable internal state — only touched under _lock ──
    private double _stressLevel;
    private double _fatigueLevel;
    private double _overloadLevel;
    private int _failureStreak;
    private long _lastPulseDurationMs;
    private DateTimeOffset _lastUpdatedUtc = DateTimeOffset.UtcNow;
    private readonly List<string> _activeFlags = new();

    // ── Recent stress source memory (bounded, under _lock) ──
    private readonly LinkedList<StressSourceEntry> _recentStressSources = new();
    private const int MaxRecentStressSources = 50;

    // ── Small recent event cache for observability (replaces former primary event log) ──
    private readonly LinkedList<AutonomicEventRecord> _recentEventCache = new();
    private const int MaxRecentEventCacheCount = 200;

    // ── Tuning constants ──
    private const double StressDecayRate = 0.85;
    private const double FatigueDecayRate = 0.90;
    private const double OverloadDecayRate = 0.80;
    private const double StressInjectionScale = 0.25;
    private const double FatiguePerFailure = 0.15;
    private const double OverloadThreshold = 0.7;
    private const double BreathlessThreshold = 0.85;
    private const int BreathlessFailureStreak = 3;
    private const long SlowPulseThresholdMs = 15_000;

    public Homeostasis(IAlephBus bus, ILogger<Homeostasis> logger)
    {
        _bus = bus;
        _logger = logger;
    }

    // ─── IHomeostasis ────────────────────────────────────────────────

    public HomeostasisSnapshot GetSnapshot()
    {
        lock (_lock)
        {
            return new HomeostasisSnapshot
            {
                StressLevel = _stressLevel,
                FatigueLevel = _fatigueLevel,
                OverloadLevel = _overloadLevel,
                FailureStreak = _failureStreak,
                LastPulseDurationMs = _lastPulseDurationMs,
                LastUpdatedUtc = _lastUpdatedUtc,
                ActiveFlags = _activeFlags.ToList().AsReadOnly(),
                RecentStressSources = _recentStressSources.Select(s => s.Source).Distinct().ToList().AsReadOnly()
            };
        }
    }

    public void Ingest(PulseEnvelope envelope)
    {
        if (envelope is null) return;

        AutonomicStressEvent? busEvent = null;

        lock (_lock)
        {
            switch (envelope.Kind)
            {
                case PulseKind.ExternalStress:
                    ApplyExternalStressUnsafe(envelope);
                    // Build bus event inside lock so we capture current state
                    busEvent = new AutonomicStressEvent
                    {
                        OccurredAtUtc = DateTimeOffset.UtcNow,
                        Source = envelope.Source,
                        Kind = "external_stress",
                        Severity = envelope.Severity,
                        Tags = envelope.Tags,
                        Message = envelope.Message,
                        Metrics = envelope.Metrics,
                        StressLevel = _stressLevel,
                        FatigueLevel = _fatigueLevel,
                        OverloadLevel = _overloadLevel,
                        FailureStreak = _failureStreak
                    };
                    break;

                case PulseKind.SystemEvent:
                    if (envelope.Tags is not null)
                    {
                        foreach (var tag in envelope.Tags)
                        {
                            if (!_activeFlags.Contains(tag))
                                _activeFlags.Add(tag);
                        }
                    }
                    break;
            }

            _lastUpdatedUtc = DateTimeOffset.UtcNow;
        }

        // Publish outside lock — non-blocking fire-and-forget
        if (busEvent is not null)
            _ = _bus.PublishAsync(busEvent);
    }

    public void RecordPulse(PulseEnvelope report)
    {
        if (report is null) return;

        HeartbeatPulseEvent? busEvent = null;

        lock (_lock)
        {
            var durationMs = report.Metrics?.GetValueOrDefault("durationMs") ?? 0;
            var success = (report.Metrics?.GetValueOrDefault("success") ?? 0) > 0.5;

            _lastPulseDurationMs = (long)durationMs;

            if (success)
            {
                _failureStreak = 0;
                _fatigueLevel = Math.Clamp(_fatigueLevel * FatigueDecayRate, 0, 1);
            }
            else
            {
                _failureStreak++;
                _fatigueLevel = Math.Clamp(_fatigueLevel + FatiguePerFailure, 0, 1);
            }

            // Overload tracks how slow pulses are relative to threshold
            if (durationMs > SlowPulseThresholdMs)
            {
                var overloadIncrement = Math.Min((durationMs - SlowPulseThresholdMs) / (double)SlowPulseThresholdMs, 0.3);
                _overloadLevel = Math.Clamp(_overloadLevel + overloadIncrement, 0, 1);
            }
            else
            {
                _overloadLevel = Math.Clamp(_overloadLevel * OverloadDecayRate, 0, 1);
            }

            // Natural decay of stress each pulse
            _stressLevel = Math.Clamp(_stressLevel * StressDecayRate, 0, 1);

            _lastUpdatedUtc = DateTimeOffset.UtcNow;

            // Update flags
            _activeFlags.RemoveAll(f => f == "overloaded" || f == "breathless");
            if (IsOverloadedUnsafe) _activeFlags.Add("overloaded");
            if (IsBreathlessUnsafe) _activeFlags.Add("breathless");

            // Record pulse in recent cache for observability
            CacheEventUnsafe(new AutonomicEventRecord
            {
                TimestampUtc = DateTimeOffset.UtcNow,
                Source = report.Source,
                Kind = "pulse_result",
                Severity = report.Severity,
                Message = success ? "Pulse succeeded" : $"Pulse failed: {report.Message}",
                Metrics = report.Metrics,
                Tags = report.Tags
            });

            // Build bus event inside lock to capture current state
            busEvent = new HeartbeatPulseEvent
            {
                OccurredAtUtc = DateTimeOffset.UtcNow,
                Source = report.Source,
                Kind = "heartbeat_pulse",
                Severity = report.Severity,
                Tags = report.Tags,
                DurationMs = (long)durationMs,
                Success = success,
                Message = success ? "Pulse succeeded" : $"Pulse failed: {report.Message}",
                Metrics = report.Metrics,
                StressLevel = _stressLevel,
                FatigueLevel = _fatigueLevel,
                OverloadLevel = _overloadLevel,
                FailureStreak = _failureStreak
            };

            _logger.LogDebug(
                "[Homeostasis] Pulse recorded. Duration={DurationMs}ms, Success={Success}, " +
                "Stress={Stress:F3}, Fatigue={Fatigue:F3}, Overload={Overload:F3}, FailStreak={FailStreak}",
                (long)durationMs, success, _stressLevel, _fatigueLevel, _overloadLevel, _failureStreak);
        }

        // Publish outside lock — non-blocking fire-and-forget
        _ = _bus.PublishAsync(busEvent);
    }

    public bool IsOverloaded
    {
        get { lock (_lock) { return IsOverloadedUnsafe; } }
    }

    public bool IsBreathless
    {
        get { lock (_lock) { return IsBreathlessUnsafe; } }
    }

    // ─── IStressInjector (write-only pathway) ────────────────────────

    public StressInjectionReceipt InjectStress(PulseEnvelope envelope)
    {
        if (envelope is null)
            return StressInjectionReceipt.Rejected("unknown", PulseSeverity.Normal, "Envelope was null.");

        if (envelope.Kind != PulseKind.ExternalStress)
            return StressInjectionReceipt.Rejected(envelope.Source, envelope.Severity,
                $"Only ExternalStress envelopes accepted. Got: {envelope.Kind}");

        if (string.IsNullOrWhiteSpace(envelope.Source))
            return StressInjectionReceipt.Rejected("unknown", envelope.Severity, "Source must be provided.");

        AutonomicStressEvent busEvent;

        lock (_lock)
        {
            ApplyExternalStressUnsafe(envelope);
            _lastUpdatedUtc = DateTimeOffset.UtcNow;

            busEvent = new AutonomicStressEvent
            {
                OccurredAtUtc = DateTimeOffset.UtcNow,
                Source = envelope.Source,
                Kind = "external_stress",
                Severity = envelope.Severity,
                Tags = envelope.Tags,
                Message = envelope.Message,
                Metrics = envelope.Metrics,
                StressLevel = _stressLevel,
                FatigueLevel = _fatigueLevel,
                OverloadLevel = _overloadLevel,
                FailureStreak = _failureStreak
            };
        }

        _logger.LogInformation(
            "[Homeostasis] Stress injected via IStressInjector. Source={Source}, Severity={Severity}",
            envelope.Source, envelope.Severity);

        // Publish outside lock — non-blocking fire-and-forget
        _ = _bus.PublishAsync(busEvent);

        return StressInjectionReceipt.Ok(envelope.Source, envelope.Severity);
    }

    // ─── Recent Event Cache (small in-memory observability window) ──

    /// <summary>
    /// Returns a snapshot of recent autonomic events, newest first.
    /// This is a small cache for observability; durable history lives in the Kidneys (DB).
    /// </summary>
    public IReadOnlyList<AutonomicEventRecord> GetRecentEvents(int maxCount = 100)
    {
        lock (_lock)
        {
            return _recentEventCache.Take(Math.Min(maxCount, _recentEventCache.Count)).ToList().AsReadOnly();
        }
    }

    // ─── Private helpers (must be called under _lock) ───────────────

    private void ApplyExternalStressUnsafe(PulseEnvelope envelope)
    {
        var intensity = (double)envelope.Severity / (double)PulseSeverity.Critical;
        _stressLevel = Math.Clamp(_stressLevel + intensity * StressInjectionScale, 0, 1);

        // Track the stress source
        _recentStressSources.AddFirst(new StressSourceEntry(
            envelope.Source, envelope.Severity, DateTimeOffset.UtcNow,
            envelope.Message));

        while (_recentStressSources.Count > MaxRecentStressSources)
            _recentStressSources.RemoveLast();

        // Cache for observability
        CacheEventUnsafe(new AutonomicEventRecord
        {
            TimestampUtc = DateTimeOffset.UtcNow,
            Source = envelope.Source,
            Kind = "external_stress",
            Severity = envelope.Severity,
            Message = envelope.Message ?? $"Stress injection severity={envelope.Severity}",
            Metrics = envelope.Metrics,
            Tags = envelope.Tags
        });

        _logger.LogDebug(
            "[Homeostasis] External stress applied from {Source}. Severity={Severity}, StressLevel={Stress:F3}",
            envelope.Source, envelope.Severity, _stressLevel);
    }

    private void CacheEventUnsafe(AutonomicEventRecord evt)
    {
        _recentEventCache.AddFirst(evt);

        while (_recentEventCache.Count > MaxRecentEventCacheCount)
            _recentEventCache.RemoveLast();
    }

    private bool IsOverloadedUnsafe => _overloadLevel >= OverloadThreshold;

    private bool IsBreathlessUnsafe =>
        _fatigueLevel >= BreathlessThreshold || _failureStreak >= BreathlessFailureStreak;
}

/// <summary>
/// Internal record tracking a recent stress injection source.
/// Used for observability — included in snapshots.
/// </summary>
internal sealed record StressSourceEntry(
    string Source,
    PulseSeverity Severity,
    DateTimeOffset TimestampUtc,
    string? Message);
