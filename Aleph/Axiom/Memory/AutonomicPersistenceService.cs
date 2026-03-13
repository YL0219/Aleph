using System.Text.Json;
using System.Threading.Channels;
using Microsoft.EntityFrameworkCore;

namespace Aleph;

/// <summary>
/// The Kidneys — a background persistence consumer that subscribes to the AlephBus
/// and writes autonomic/heartbeat events to SQLite in batches.
/// Filters for AutonomicStressEvent and HeartbeatPulseEvent only.
/// Applies retention: max 14 days, max 10,000 rows.
/// </summary>
public sealed class AutonomicPersistenceService : BackgroundService
{
    /// <summary>Maximum number of persisted autonomic events.</summary>
    public const int MaxRetentionCount = 10_000;

    /// <summary>Maximum age of persisted autonomic events.</summary>
    public static readonly TimeSpan MaxRetentionAge = TimeSpan.FromDays(14);

    private const int BatchSize = 50;
    private const int RetentionCheckInterval = 100; // run retention every N events written

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = false
    };

    private readonly IAlephBus _bus;
    private readonly IDbContextFactory<AppDbContext> _dbFactory;
    private readonly ILogger<AutonomicPersistenceService> _logger;

    private int _totalWritten;

    public AutonomicPersistenceService(
        IAlephBus bus,
        IDbContextFactory<AppDbContext> dbFactory,
        ILogger<AutonomicPersistenceService> logger)
    {
        _bus = bus;
        _dbFactory = dbFactory;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("[Kidneys] Autonomic persistence service started.");

        // Ensure the AutonomicEvents table exists (safe for DBs created without migrations)
        await EnsureTableExistsAsync(stoppingToken);

        // Subscribe only to stress + heartbeat events
        var reader = _bus.Subscribe("Kidneys", evt =>
            evt is AutonomicStressEvent or HeartbeatPulseEvent);

        var batch = new List<AutonomicEvent>(BatchSize);

        try
        {
            await foreach (var evt in reader.ReadAllAsync(stoppingToken))
            {
                var entity = MapToEntity(evt);
                if (entity is null) continue;

                batch.Add(entity);

                if (batch.Count >= BatchSize)
                {
                    await FlushBatchAsync(batch, stoppingToken);
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Shutting down — flush remaining
        }

        // Flush any remaining events
        if (batch.Count > 0)
        {
            try
            {
                await FlushBatchAsync(batch, CancellationToken.None);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[Kidneys] Failed to flush final batch on shutdown.");
            }
        }

        _logger.LogInformation("[Kidneys] Autonomic persistence service stopped. Total events written: {Total}", _totalWritten);
    }

    private async Task FlushBatchAsync(List<AutonomicEvent> batch, CancellationToken ct)
    {
        try
        {
            await using var db = await _dbFactory.CreateDbContextAsync(ct);

            db.AutonomicEvents.AddRange(batch);
            await db.SaveChangesAsync(ct);

            _totalWritten += batch.Count;
            _logger.LogDebug("[Kidneys] Flushed {Count} autonomic events to DB. Total: {Total}", batch.Count, _totalWritten);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Kidneys] Failed to flush {Count} events.", batch.Count);
        }
        finally
        {
            batch.Clear();
        }

        // Periodic retention enforcement
        if (_totalWritten % RetentionCheckInterval < BatchSize)
        {
            await EnforceRetentionAsync(ct);
        }
    }

    private async Task EnforceRetentionAsync(CancellationToken ct)
    {
        try
        {
            await using var db = await _dbFactory.CreateDbContextAsync(ct);

            // Enforce max age
            var cutoff = DateTime.UtcNow - MaxRetentionAge;
            var aged = await db.AutonomicEvents
                .Where(e => e.OccurredAtUtc < cutoff)
                .ExecuteDeleteAsync(ct);

            if (aged > 0)
                _logger.LogInformation("[Kidneys] Retention: pruned {Count} events older than {Days} days.", aged, MaxRetentionAge.TotalDays);

            // Enforce max count
            var totalCount = await db.AutonomicEvents.CountAsync(ct);
            if (totalCount > MaxRetentionCount)
            {
                var excess = totalCount - MaxRetentionCount;
                var oldestIds = await db.AutonomicEvents
                    .OrderBy(e => e.OccurredAtUtc)
                    .Take(excess)
                    .Select(e => e.Id)
                    .ToListAsync(ct);

                if (oldestIds.Count > 0)
                {
                    await db.AutonomicEvents
                        .Where(e => oldestIds.Contains(e.Id))
                        .ExecuteDeleteAsync(ct);

                    _logger.LogInformation("[Kidneys] Retention: pruned {Count} excess events (cap={Cap}).", oldestIds.Count, MaxRetentionCount);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[Kidneys] Retention enforcement failed (non-fatal).");
        }
    }

    private async Task EnsureTableExistsAsync(CancellationToken ct)
    {
        try
        {
            await using var db = await _dbFactory.CreateDbContextAsync(ct);
            await db.Database.ExecuteSqlRawAsync("""
                CREATE TABLE IF NOT EXISTS AutonomicEvents (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    OccurredAtUtc TEXT NOT NULL DEFAULT (datetime('now')),
                    Source TEXT NOT NULL DEFAULT '',
                    Kind TEXT NOT NULL DEFAULT '',
                    Severity INTEGER NOT NULL DEFAULT 0,
                    Message TEXT,
                    MetricsJson TEXT,
                    TagsJson TEXT,
                    StressLevel REAL NOT NULL DEFAULT 0,
                    FatigueLevel REAL NOT NULL DEFAULT 0,
                    OverloadLevel REAL NOT NULL DEFAULT 0,
                    FailureStreak INTEGER NOT NULL DEFAULT 0,
                    CorrelationId TEXT
                );
                CREATE INDEX IF NOT EXISTS IX_AutonomicEvents_OccurredAtUtc ON AutonomicEvents(OccurredAtUtc);
                CREATE INDEX IF NOT EXISTS IX_AutonomicEvents_Kind ON AutonomicEvents(Kind);
                """, ct);

            _logger.LogInformation("[Kidneys] AutonomicEvents table ensured.");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Kidneys] Failed to ensure AutonomicEvents table.");
        }
    }

    private static AutonomicEvent? MapToEntity(AlephEvent evt)
    {
        return evt switch
        {
            AutonomicStressEvent stress => new AutonomicEvent
            {
                OccurredAtUtc = stress.OccurredAtUtc.UtcDateTime,
                Source = stress.Source,
                Kind = stress.Kind,
                Severity = (int)stress.Severity,
                Message = stress.Message,
                MetricsJson = stress.Metrics is not null ? JsonSerializer.Serialize(stress.Metrics, JsonOpts) : null,
                TagsJson = stress.Tags is not null ? JsonSerializer.Serialize(stress.Tags, JsonOpts) : null,
                StressLevel = stress.StressLevel,
                FatigueLevel = stress.FatigueLevel,
                OverloadLevel = stress.OverloadLevel,
                FailureStreak = stress.FailureStreak,
                CorrelationId = stress.CorrelationId
            },

            HeartbeatPulseEvent pulse => new AutonomicEvent
            {
                OccurredAtUtc = pulse.OccurredAtUtc.UtcDateTime,
                Source = pulse.Source,
                Kind = pulse.Kind,
                Severity = (int)pulse.Severity,
                Message = pulse.Message,
                MetricsJson = pulse.Metrics is not null ? JsonSerializer.Serialize(pulse.Metrics, JsonOpts) : null,
                TagsJson = pulse.Tags is not null ? JsonSerializer.Serialize(pulse.Tags, JsonOpts) : null,
                StressLevel = pulse.StressLevel,
                FatigueLevel = pulse.FatigueLevel,
                OverloadLevel = pulse.OverloadLevel,
                FailureStreak = pulse.FailureStreak,
                CorrelationId = pulse.CorrelationId
            },

            _ => null
        };
    }
}
