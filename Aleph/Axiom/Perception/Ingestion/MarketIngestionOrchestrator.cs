// CONTRACT / INVARIANTS
// - Scoped callable cycle runner: performs ONE ingestion cycle on demand.
// - No internal timers, no while-loop, no hosted behavior.
// - Heartbeat resolves this via IServiceScopeFactory each pulse.
// - Active symbols = portfolio holdings (IsOpen or Quantity > 0) UNION watchlist (IsActive).
// - Batches of 10 symbols per ingestion invocation.
// - DB writes and Python execution are routed through IAxiom.MarketIngestion.
// - If Python is unavailable, logs a warning and returns (does NOT crash).
// - Parquet path contract: data_lake/market/ohlcv/symbol=<SYM>/interval=<INTERVAL>/latest.parquet

namespace Aleph;

/// <summary>
/// Interface for the demoted ingestion cycle runner.
/// Resolved as Scoped via IServiceScopeFactory from HeartbeatService.
/// </summary>
public interface IMarketIngestionCycle
{
    /// <summary>Run a single ingestion cycle for all active symbols.</summary>
    Task RunCycleAsync(CancellationToken ct);
}

/// <summary>
/// Performs one ingestion cycle on demand. No longer a BackgroundService.
/// </summary>
public class MarketIngestionOrchestrator : IMarketIngestionCycle
{
    private const int BatchSize = 10;
    private const int LookbackDays = 365;
    private const string DefaultInterval = "1d";
    private const string OutRoot = "data_lake/market/ohlcv";

    private readonly IAxiom _axiom;
    private readonly IAlephBus _bus;
    private readonly IMarketStressDetector _stressDetector;
    private readonly ILogger<MarketIngestionOrchestrator> _logger;

    public MarketIngestionOrchestrator(
        IAxiom axiom,
        IAlephBus bus,
        IMarketStressDetector stressDetector,
        ILogger<MarketIngestionOrchestrator> logger)
    {
        _axiom = axiom;
        _bus = bus;
        _stressDetector = stressDetector;
        _logger = logger;
    }

    public async Task RunCycleAsync(CancellationToken ct)
    {
        if (!_axiom.MarketIngestion.IsPythonAvailable)
        {
            _logger.LogWarning(
                "[Ingestion] Python not available. " +
                "Ingestion cycle skipped. Run setup_venv.ps1 to enable ingestion.");
            return;
        }

        _logger.LogInformation("[Ingestion] Starting ingestion cycle...");

        try
        {
            var symbols = await _axiom.MarketIngestion.GetActiveSymbolsAsync(ct);
            if (symbols.Count == 0)
            {
                _logger.LogInformation("[Ingestion] No active symbols to ingest. Skipping.");
                return;
            }

            _logger.LogInformation("[Ingestion] Active symbols ({Count}): {Symbols}",
                symbols.Count, string.Join(", ", symbols));

            var batches = symbols.Chunk(BatchSize).ToList();
            for (int i = 0; i < batches.Count; i++)
            {
                ct.ThrowIfCancellationRequested();

                var batchSymbols = batches[i].ToList();
                _logger.LogInformation("[Ingestion] Processing batch {Current}/{Total}: {Symbols}",
                    i + 1, batches.Count, string.Join(", ", batchSymbols));

                var runResult = await _axiom.MarketIngestion.RunIngestionBatchAsync(
                    batchSymbols, DefaultInterval, LookbackDays, OutRoot, ct);

                if (runResult.Report is not null)
                {
                    _logger.LogInformation(
                        "[Ingestion] Report received: jobId={JobId}, duration={Duration}ms, results={Count}",
                        runResult.Report.JobId,
                        runResult.Report.DurationMs,
                        runResult.Report.Results.Count);
                }

                await _axiom.MarketIngestion.ApplyIngestionBatchAsync(
                    batchSymbols, DefaultInterval, runResult, ct);

                // ── Publish MarketDataEvent for each ingested symbol ──
                // This connects the background ingestion vein to the Liver and ML Cortex.
                await PublishIngestionEventsAsync(runResult, ct);
            }

            // ── Perception refresh: macro proxies, calendar, headlines ──
            _logger.LogInformation("[Ingestion] Running perception refresh...");
            try
            {
                await RunPerceptionRefreshAsync(ct);
            }
            catch (Exception percEx) when (percEx is not OperationCanceledException)
            {
                // Perception failure must NOT crash the ingestion cycle
                _logger.LogWarning(percEx, "[Ingestion] Perception refresh failed (non-fatal).");
            }

            _logger.LogInformation("[Ingestion] Cycle complete. Running reflexive stress evaluation...");

            // Reflexive stress detection — runs after fresh data is available
            try
            {
                await _stressDetector.EvaluateAsync(ct);
            }
            catch (Exception stressEx) when (stressEx is not OperationCanceledException)
            {
                // Stress detection failure must NOT crash the ingestion cycle
                _logger.LogWarning(stressEx, "[Ingestion] Reflexive stress evaluation failed (non-fatal).");
            }
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("[Ingestion] Cycle cancelled.");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Ingestion] Cycle failed.");
            throw; // Let HeartbeatService catch and track in Homeostasis
        }
    }

    /// <summary>
    /// Run perception ingest and publish a PerceptionRefreshEvent.
    /// </summary>
    private async Task RunPerceptionRefreshAsync(CancellationToken ct)
    {
        var result = await _axiom.Perception.RunIngestAsync(ct: ct);

        var evt = new PerceptionRefreshEvent
        {
            OccurredAtUtc = DateTimeOffset.UtcNow,
            Source = "Ingestion",
            Kind = "perception_refresh",
            Severity = result.Success ? PulseSeverity.Normal : PulseSeverity.Warning,
            Success = result.Success,
            DurationMs = result.DurationMs,
            ProxiesSucceeded = result.ProxiesSucceeded,
            ProxiesTotal = result.ProxiesTotal,
            CalendarOk = result.CalendarOk,
            CalendarProvider = result.CalendarProvider,
            CalendarEventCount = result.CalendarEventCount,
            HeadlinesOk = result.HeadlinesOk,
            HeadlinesProvider = result.HeadlinesProvider,
            HeadlineCount = result.HeadlineCount,
            ManifestPath = result.ManifestPath,
            ErrorMessage = result.ErrorMessage,
        };

        await _bus.PublishAsync(evt, ct);

        if (result.Success)
        {
            _logger.LogInformation(
                "[Ingestion] Perception refresh complete: proxies={Proxies}/{Total}, calendar={Cal}, headlines={Head}",
                result.ProxiesSucceeded, result.ProxiesTotal,
                result.CalendarOk ? result.CalendarEventCount.ToString() + " events" : "failed",
                result.HeadlinesOk ? result.HeadlineCount.ToString() + " items" : "failed");
        }
        else
        {
            _logger.LogWarning("[Ingestion] Perception refresh reported failure: {Error}", result.ErrorMessage);
        }
    }

    /// <summary>
    /// Publish a MarketDataEvent for each successfully ingested symbol.
    /// This is the missing vein — connects background ingestion to the Liver and ML Cortex.
    /// </summary>
    private async Task PublishIngestionEventsAsync(MarketIngestionRunResult runResult, CancellationToken ct)
    {
        if (runResult.Report?.Results is null)
            return;

        foreach (var result in runResult.Report.Results)
        {
            var evt = new MarketDataEvent
            {
                OccurredAtUtc = DateTimeOffset.UtcNow,
                Source = "Ingestion",
                Kind = "market_data",
                Severity = result.IsSuccess ? PulseSeverity.Normal : PulseSeverity.Warning,
                Symbol = result.Symbol,
                Interval = string.IsNullOrWhiteSpace(result.Interval) ? DefaultInterval : result.Interval,
                Success = result.IsSuccess,
                SourceKind = "background_ingestion",
                RowsWritten = result.IsSuccess ? result.RowsWritten : null,
                ParquetPath = result.IsSuccess ? result.ParquetPath : null,
                ErrorMessage = result.Error?.Message,
            };

            await _bus.PublishAsync(evt, ct);

            if (result.IsSuccess)
            {
                _logger.LogDebug(
                    "[Ingestion] Published MarketDataEvent for {Symbol}/{Interval} ({Rows} rows).",
                    result.Symbol, result.Interval, result.RowsWritten);
            }
        }
    }
}
