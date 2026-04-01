namespace Aleph;

public sealed class HeartbeatService : BackgroundService
{
    private static readonly DayOfWeek[] TradingDays =
    {
        DayOfWeek.Monday,
        DayOfWeek.Tuesday,
        DayOfWeek.Wednesday,
        DayOfWeek.Thursday,
        DayOfWeek.Friday
    };

    internal static readonly string[] MacroBasketSymbols = { "SPY", "QQQ", "TLT", "GLD" };

    private readonly IAether _aether;
    private readonly IHomeostasis _homeostasis;
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly ILogger<HeartbeatService> _logger;
    private readonly string _slowCycleTrainSymbol;

    // Base cadences (before governor adjustment)
    private readonly double _baseFastDelaySeconds;
    private readonly double _baseSlowDelaySeconds;
    private readonly TimeSpan _marketWindowStartUtc;
    private readonly TimeSpan _marketWindowEndUtc;

    // Governor limits
    private const double MinDelaySeconds = 10;
    private const double MaxDelaySeconds = 600; // 10 minutes cap
    private const double GovernorSlowdownMultiplier = 2.0;
    private const double StressSpeedupFactor = 0.7; // faster when stressed (but governor can override)

    public HeartbeatService(
        IAether aether,
        IHomeostasis homeostasis,
        IServiceScopeFactory scopeFactory,
        IConfiguration configuration,
        ILogger<HeartbeatService> logger)
    {
        _aether = aether;
        _homeostasis = homeostasis;
        _scopeFactory = scopeFactory;
        _logger = logger;

        _baseFastDelaySeconds = ReadInt(configuration, "Aether:Heartbeat:FastDelaySeconds", 30, 5, 3600);
        _baseSlowDelaySeconds = ReadInt(configuration, "Aether:Heartbeat:SlowDelayMinutes", 10, 1, 720) * 60.0;
        _slowCycleTrainSymbol = ReadSymbol(configuration, "Aether:SleepCycle:Symbol", "SI=F");

        _marketWindowStartUtc = TimeSpan.FromHours(ReadInt(configuration, "Aether:Heartbeat:MarketWindowStartUtcHour", 13, 0, 23));
        _marketWindowEndUtc = TimeSpan.FromHours(ReadInt(configuration, "Aether:Heartbeat:MarketWindowEndUtcHour", 20, 0, 23));
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("[Heartbeat] Cybernetic heart started. Waiting for system warmup...");

        // Brief warmup so DI, DB migrations, etc. settle
        try { await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken); }
        catch (OperationCanceledException) { return; }

        _logger.LogInformation("[Heartbeat] First pulse initiating.");

        while (!stoppingToken.IsCancellationRequested)
        {
            var pulseStart = DateTimeOffset.UtcNow;

            try
            {
                // ── Step 1: Read homeostatic snapshot ──
                var snapshot = _homeostasis.GetSnapshot();

                // ── Step 2: Determine desired mode from stress + time window ──
                var desiredMode = DetermineDesiredMode(pulseStart, snapshot);

                // ── Step 3: Determine safe mode from fatigue / overload ──
                var safeMode = DetermineSafeMode(desiredMode, snapshot);

                _logger.LogInformation(
                    "[Heartbeat] Pulse. Desired={Desired}, Safe={Safe}, Stress={Stress:F2}, Fatigue={Fatigue:F2}, Overload={Overload:F2}, FailStreak={FS}",
                    desiredMode, safeMode, snapshot.StressLevel, snapshot.FatigueLevel, snapshot.OverloadLevel, snapshot.FailureStreak);

                // ── Step 5-6: Create scope, resolve ingestion runner, run one cycle ──
                await RunIngestionCycleAsync(stoppingToken);

                // ── Step 7: Run Aether work based on mode ──
                await RunAetherWorkAsync(safeMode, snapshot, stoppingToken);

                // ── Step 4 (applied after work): Compute actual delay ──
                var delaySeconds = ComputeGovernedDelay(safeMode, snapshot);
                var delay = TimeSpan.FromSeconds(delaySeconds);

                // ── Step 8: Measure pulse duration ──
                var pulseDuration = DateTimeOffset.UtcNow - pulseStart;

                _logger.LogInformation(
                    "[Heartbeat] Pulse complete. Duration={DurationMs}ms, Mode={Mode}, NextDelay={DelaySec:F1}s",
                    (long)pulseDuration.TotalMilliseconds, safeMode, delaySeconds);

                // ── Step 9: Report pulse health back into Homeostasis ──
                var report = PulseEnvelope.HeartbeatReport(
                    (long)pulseDuration.TotalMilliseconds,
                    success: true,
                    mode: safeMode);
                _homeostasis.RecordPulse(report);

                // ── Step 10: Delay until next beat ──
                try { await Task.Delay(delay, stoppingToken); }
                catch (OperationCanceledException) { break; }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                var pulseDuration = DateTimeOffset.UtcNow - pulseStart;

                _logger.LogError(ex, "[Heartbeat] Pulse failed after {DurationMs}ms.", (long)pulseDuration.TotalMilliseconds);

                // Report failure into Homeostasis
                var report = PulseEnvelope.HeartbeatReport(
                    (long)pulseDuration.TotalMilliseconds,
                    success: false,
                    mode: HeartbeatMode.Slow,
                    message: ex.Message);
                _homeostasis.RecordPulse(report);

                // Back off after failure — use overload-aware delay
                var backoffSeconds = Math.Min(_baseSlowDelaySeconds, MaxDelaySeconds);
                try { await Task.Delay(TimeSpan.FromSeconds(backoffSeconds), stoppingToken); }
                catch (OperationCanceledException) { break; }
            }
        }

        _logger.LogInformation("[Heartbeat] Cybernetic heart stopped.");
    }

    // ─── Governor: Desired Mode ───

    internal HeartbeatMode DetermineDesiredMode(DateTimeOffset nowUtc, HomeostasisSnapshot snapshot)
    {
        // Base mode from market window
        var baseMode = DetermineHeartbeatMode(nowUtc);

        // High stress can promote Slow → Fast (market danger = pay attention)
        if (baseMode == HeartbeatMode.Slow && snapshot.StressLevel > 0.6)
            return HeartbeatMode.Fast;

        return baseMode;
    }

    // ─── Governor: Safe Mode ───

    private static HeartbeatMode DetermineSafeMode(HeartbeatMode desired, HomeostasisSnapshot snapshot)
    {
        // If breathless or overloaded, force slow regardless of desire
        if (snapshot.FatigueLevel >= 0.85 || snapshot.FailureStreak >= 3 || snapshot.OverloadLevel >= 0.7)
            return HeartbeatMode.Slow;

        return desired;
    }

    // ─── Governor: Compute Delay ───

    private double ComputeGovernedDelay(HeartbeatMode mode, HomeostasisSnapshot snapshot)
    {
        var baseDelay = mode == HeartbeatMode.Fast ? _baseFastDelaySeconds : _baseSlowDelaySeconds;

        // Stress can speed up (but only in Fast mode and only modestly)
        if (mode == HeartbeatMode.Fast && snapshot.StressLevel > 0.3)
        {
            baseDelay *= StressSpeedupFactor;
        }

        // Fatigue / overload slow things down — governor override
        if (snapshot.FatigueLevel > 0.5 || snapshot.OverloadLevel > 0.5)
        {
            var worstLevel = Math.Max(snapshot.FatigueLevel, snapshot.OverloadLevel);
            var slowdownFactor = 1.0 + (worstLevel * GovernorSlowdownMultiplier);
            baseDelay *= slowdownFactor;
        }

        // Failure streak adds backoff
        if (snapshot.FailureStreak > 0)
        {
            baseDelay *= 1.0 + (snapshot.FailureStreak * 0.5);
        }

        return Math.Clamp(baseDelay, MinDelaySeconds, MaxDelaySeconds);
    }

    // ─── Ingestion (Scoped) ───

    private async Task RunIngestionCycleAsync(CancellationToken ct)
    {
        using var scope = _scopeFactory.CreateScope();
        var ingestion = scope.ServiceProvider.GetRequiredService<IMarketIngestionCycle>();
        await ingestion.RunCycleAsync(ct);
    }

    // ─── Aether Work ───

    private async Task RunAetherWorkAsync(HeartbeatMode mode, HomeostasisSnapshot snapshot, CancellationToken ct)
    {
        if (mode == HeartbeatMode.Fast)
        {
            await _aether.Ml.GetStatusAsync(new MlStatusRequest(), ct);
            await _aether.Macro.CheckRegimeAsync(new MacroRegimeRequest(), ct);
            _logger.LogInformation("[Heartbeat] Fast cycle Aether work completed.");
        }
        else
        {
            // In slow mode, skip heavy work if overloaded/breathless
            if (snapshot.OverloadLevel >= 0.7 || snapshot.FatigueLevel >= 0.7)
            {
                _logger.LogInformation("[Heartbeat] Slow cycle — shedding heavy work (overloaded/fatigued).");
                // Only do lightweight status check
                await _aether.Ml.GetStatusAsync(new MlStatusRequest(), ct);
            }
            else
            {
                await _aether.Ml.TrainAsync(new MlTrainRequest(_slowCycleTrainSymbol, 1), ct);
                _logger.LogInformation(
                    "[Heartbeat] Slow cycle Aether work completed (train symbol={Symbol}).",
                    _slowCycleTrainSymbol);
            }
        }
    }

    // ─── Time-Window Mode (unchanged logic) ───

    internal HeartbeatMode DetermineHeartbeatMode(DateTimeOffset nowUtc)
    {
        if (!TradingDays.Contains(nowUtc.DayOfWeek))
            return HeartbeatMode.Slow;

        var tod = nowUtc.TimeOfDay;

        if (_marketWindowStartUtc <= _marketWindowEndUtc)
        {
            return tod >= _marketWindowStartUtc && tod < _marketWindowEndUtc
                ? HeartbeatMode.Fast
                : HeartbeatMode.Slow;
        }

        var isFastOvernight = tod >= _marketWindowStartUtc || tod < _marketWindowEndUtc;
        return isFastOvernight ? HeartbeatMode.Fast : HeartbeatMode.Slow;
    }

    private static int ReadInt(IConfiguration configuration, string key, int fallback, int min, int max)
    {
        var raw = configuration[key];
        if (!int.TryParse(raw, out var parsed))
            return fallback;

        return Math.Clamp(parsed, min, max);
    }

    private static string ReadSymbol(IConfiguration configuration, string key, string fallback)
    {
        var configured = configuration[key];
        if (SymbolValidator.TryNormalize(configured, out var normalized))
            return normalized;

        return SymbolValidator.TryNormalize(fallback, out var fallbackNormalized)
            ? fallbackNormalized
            : "SI=F";
    }
}

public enum HeartbeatMode
{
    Fast = 0,
    Slow = 1
}
