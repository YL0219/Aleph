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

    private readonly IAether _aether;
    private readonly ILogger<HeartbeatService> _logger;

    private readonly TimeSpan _fastDelay;
    private readonly TimeSpan _slowDelay;
    private readonly TimeSpan _marketWindowStartUtc;
    private readonly TimeSpan _marketWindowEndUtc;

    public HeartbeatService(
        IAether aether,
        IConfiguration configuration,
        ILogger<HeartbeatService> logger)
    {
        _aether = aether;
        _logger = logger;

        _fastDelay = TimeSpan.FromSeconds(ReadInt(configuration, "Aether:Heartbeat:FastDelaySeconds", 30, 5, 3600));
        _slowDelay = TimeSpan.FromMinutes(ReadInt(configuration, "Aether:Heartbeat:SlowDelayMinutes", 10, 1, 720));

        _marketWindowStartUtc = TimeSpan.FromHours(ReadInt(configuration, "Aether:Heartbeat:MarketWindowStartUtcHour", 13, 0, 23));
        _marketWindowEndUtc = TimeSpan.FromHours(ReadInt(configuration, "Aether:Heartbeat:MarketWindowEndUtcHour", 20, 0, 23));
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("[Heartbeat] Service started.");

        while (!stoppingToken.IsCancellationRequested)
        {
            var nowUtc = DateTimeOffset.UtcNow;
            var mode = DetermineHeartbeatMode(nowUtc);

            _logger.LogInformation("[Heartbeat] Pulse triggered. Mode: {Mode}", mode);

            try
            {
                if (mode == HeartbeatMode.Fast)
                {
                    await _aether.Ml.GetStatusAsync(new MlStatusRequest(), stoppingToken);
                    await _aether.Macro.CheckRegimeAsync(new MacroRegimeRequest(), stoppingToken);
                    _logger.LogInformation("[Heartbeat] Fast cycle completed.");
                }
                else
                {
                    await _aether.Ml.TrainAsync(new MlTrainRequest("SPY", 1), stoppingToken);
                    _logger.LogInformation("[Heartbeat] Slow cycle completed.");
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[Heartbeat] Cycle failed.");
            }

            var delay = mode == HeartbeatMode.Fast ? _fastDelay : _slowDelay;
            try
            {
                await Task.Delay(delay, stoppingToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }

        _logger.LogInformation("[Heartbeat] Service stopped.");
    }

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
}

public enum HeartbeatMode
{
    Fast = 0,
    Slow = 1
}
