using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Globalization;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Caching.Memory;

namespace Aleph;

public sealed class Axiom : IAxiom
{
    private static readonly Regex SkillNamePattern =
        new("^[a-z][a-z0-9_]{0,63}$", RegexOptions.Compiled);

    private readonly IDbContextFactory<AppDbContext> _dbFactory;
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly PythonDispatcherService _pythonDispatcher;
    private readonly IMemoryCache _cache;
    private readonly McpToolInvoker _mcpInvoker;
    private readonly McpToolSchemaAdapter _mcpSchemaAdapter;
    private readonly ISkillRegistry _skillRegistry;
    private readonly IAlephBus _bus;
    private readonly IHostEnvironment _env;
    private readonly ILogger<Axiom> _logger;

    public IAxiom.IPythonRouter Python { get; }
    public IAxiom.IMarketGateway Market { get; }
    public IAxiom.IMarketIngestionGateway MarketIngestion { get; }
    public IAxiom.IMcpGateway Mcp { get; }
    public IAxiom.ITradeGateway Trades { get; }
    public IAxiom.IChatGateway Chat { get; }
    public IAxiom.IToolRunGateway ToolRuns { get; }
    public IAxiom.ISkillGateway Skills { get; }
    public IAxiom.IPerceptionGateway Perception { get; }

    public Axiom(
        IDbContextFactory<AppDbContext> dbFactory,
        IServiceScopeFactory scopeFactory,
        PythonDispatcherService pythonDispatcher,
        IMemoryCache cache,
        McpToolInvoker mcpInvoker,
        McpToolSchemaAdapter mcpSchemaAdapter,
        ISkillRegistry skillRegistry,
        IAlephBus bus,
        IHostEnvironment env,
        ILogger<Axiom> logger)
    {
        _dbFactory = dbFactory;
        _scopeFactory = scopeFactory;
        _pythonDispatcher = pythonDispatcher;
        _cache = cache;
        _mcpInvoker = mcpInvoker;
        _mcpSchemaAdapter = mcpSchemaAdapter;
        _skillRegistry = skillRegistry;
        _bus = bus;
        _env = env;
        _logger = logger;

        Python = new PythonRouterGateway(this);
        Market = new MarketGateway(this);
        MarketIngestion = new MarketIngestionGateway(this);
        Mcp = new McpGateway(this);
        Trades = new TradeGateway(this);
        Chat = new ChatGateway(this);
        ToolRuns = new ToolRunGateway(this);
        Skills = new SkillGateway(this);
        Perception = new PerceptionGateway(this);
    }

    private sealed class PythonRouterGateway : IAxiom.IPythonRouter
    {
        private readonly Axiom _root;

        public PythonRouterGateway(Axiom root)
        {
            _root = root;
        }

        public async Task<PythonRouteResult> RunJsonAsync(
            string domain,
            string action,
            IReadOnlyList<string> arguments,
            int timeoutMs,
            CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(domain))
                throw new ArgumentException("Domain is required.", nameof(domain));

            if (string.IsNullOrWhiteSpace(action))
                throw new ArgumentException("Action is required.", nameof(action));

            if (timeoutMs <= 0)
                throw new ArgumentOutOfRangeException(nameof(timeoutMs), "Timeout must be > 0.");

            var safeArgs = arguments ?? Array.Empty<string>();
            _root._logger.LogDebug(
                "[Axiom] Python route {Domain}/{Action} with {ArgCount} args.",
                domain,
                action,
                safeArgs.Count);
            var result = await _root._pythonDispatcher.RunAsync(domain, action, safeArgs, timeoutMs, ct);

            return new PythonRouteResult(
                result.Success,
                result.Stdout,
                result.Stderr,
                result.ExitCode,
                result.TimedOut);
        }
    }

    private sealed class MarketGateway : IAxiom.IMarketGateway
    {
        private static readonly TimeSpan CacheTtl = TimeSpan.FromSeconds(10);
        private const int PythonTimeoutMs = 30_000;
        private const int IngestionTimeoutMs = 90_000;
        private const int ParquetReadTimeoutMs = 30_000;
        private const string DataLakeRelativePath = "data_lake/market/ohlcv";

        /// <summary>Staleness threshold — local data older than this triggers a live refresh.</summary>
        private static readonly TimeSpan StalenessThreshold = TimeSpan.FromHours(18);

        /// <summary>Per-symbol+interval fetch gate to prevent duplicate concurrent live fetches.</summary>
        private readonly ConcurrentDictionary<string, SemaphoreSlim> _fetchGates = new();

        private readonly Axiom _root;

        public MarketGateway(Axiom root)
        {
            _root = root;
        }

        public async Task<MarketQuoteFetchResult> GetQuoteAsync(
            string normalizedSymbol,
            CancellationToken ct = default)
        {
            if (!SymbolValidator.TryNormalize(normalizedSymbol, out var symbol))
            {
                return new MarketQuoteFetchResult(false, null, "Invalid symbol.");
            }

            string cacheKey = $"quote:{symbol}";
            if (_root._cache.TryGetValue(cacheKey, out MarketQuoteDto? cached) && cached is not null)
            {
                _root._logger.LogDebug("[Market] Cache hit: {CacheKey}", cacheKey);
                return new MarketQuoteFetchResult(true, cached, null);
            }

            var routeResult = await RunFetcherAsync(
                "fetch-quote",
                new[] { "--symbol", symbol },
                ct);
            if (!routeResult.Success)
            {
                return new MarketQuoteFetchResult(false, null, routeResult.ErrorMessage);
            }

            try
            {
                using var doc = JsonDocument.Parse(routeResult.StdoutJson);
                var root = doc.RootElement;

                if (!root.TryGetProperty("ok", out var okProp) || !okProp.GetBoolean())
                {
                    string pyError = root.TryGetProperty("error", out var errProp)
                        ? errProp.GetString() ?? "Unknown"
                        : "Unknown";
                    return new MarketQuoteFetchResult(false, null, pyError);
                }

                var quote = new MarketQuoteDto(
                    root.GetProperty("symbol").GetString() ?? symbol,
                    root.GetProperty("price").GetDouble(),
                    root.GetProperty("timestampUtc").GetString() ?? string.Empty);

                _root._cache.Set(cacheKey, quote, CacheTtl);
                return new MarketQuoteFetchResult(true, quote, null);
            }
            catch (JsonException ex)
            {
                _root._logger.LogWarning(ex, "[Market] JSON parse error for quote {Symbol}", symbol);
                return new MarketQuoteFetchResult(false, null, "Invalid response from data fetcher.");
            }
        }

        public async Task<MarketCandlesFetchResult> GetCandlesAsync(
            MarketCandlesQuery query,
            CancellationToken ct = default)
        {
            if (query is null)
                throw new ArgumentNullException(nameof(query));

            if (!SymbolValidator.TryNormalize(query.Symbol, out var symbol))
            {
                return new MarketCandlesFetchResult(false, null, "Invalid symbol.");
            }

            string tf = query.Tf.ToLowerInvariant();
            string range = query.Range.ToLowerInvariant();
            string? to = query.To;

            string cacheKey = $"candles:{symbol}:{tf}:{range}:{query.Limit}:{to ?? "latest"}";
            if (_root._cache.TryGetValue(cacheKey, out MarketCandlesDto? cached) && cached is not null)
            {
                _root._logger.LogDebug("[Market] Cache hit: {CacheKey}", cacheKey);
                return new MarketCandlesFetchResult(true, cached, null);
            }

            var args = new List<string>
            {
                "--symbol", symbol,
                "--tf", tf,
                "--range", range,
                "--limit", query.Limit.ToString()
            };

            if (to is not null)
            {
                args.Add("--to");
                args.Add(to);
            }

            var routeResult = await RunFetcherAsync("fetch-candles", args, ct);
            if (!routeResult.Success)
            {
                return new MarketCandlesFetchResult(false, null, routeResult.ErrorMessage);
            }

            try
            {
                using var doc = JsonDocument.Parse(routeResult.StdoutJson);
                var root = doc.RootElement;

                if (!root.TryGetProperty("ok", out var okProp) || !okProp.GetBoolean())
                {
                    string pyError = root.TryGetProperty("error", out var errProp)
                        ? errProp.GetString() ?? "Unknown"
                        : "Unknown";
                    return new MarketCandlesFetchResult(false, null, pyError);
                }

                var candles = new MarketCandlesDto(
                    root.GetProperty("symbol").GetString() ?? symbol,
                    root.GetProperty("tf").GetString() ?? tf,
                    root.GetProperty("candles").Clone(),
                    root.TryGetProperty("nextTo", out var nextToProp) && nextToProp.ValueKind != JsonValueKind.Null
                        ? nextToProp.GetInt64()
                        : (long?)null);

                _root._cache.Set(cacheKey, candles, CacheTtl);
                return new MarketCandlesFetchResult(true, candles, null);
            }
            catch (JsonException ex)
            {
                _root._logger.LogWarning(ex, "[Market] JSON parse error for candles {Symbol}", symbol);
                return new MarketCandlesFetchResult(false, null, "Invalid response from data fetcher.");
            }
        }

        // ─── On-Demand Perception ────────────────────────────────────

        public async Task<MarketPerceptionResult> PerceiveQuoteAsync(
            string symbol,
            CancellationToken ct = default)
        {
            if (!SymbolValidator.TryNormalize(symbol, out var normalized))
                return new MarketPerceptionResult
                {
                    Success = false,
                    Symbol = symbol ?? "",
                    HistorySource = MarketHistorySource.Missing,
                    ErrorMessage = "Invalid symbol."
                };

            // Try live quote
            var quoteResult = await GetQuoteAsync(normalized, ct);
            if (quoteResult.Success && quoteResult.Quote is not null)
            {
                // Publish market data event
                _ = _root._bus.PublishAsync(new MarketDataEvent
                {
                    OccurredAtUtc = DateTimeOffset.UtcNow,
                    Source = "Axiom.Market",
                    Kind = "market_perception",
                    Severity = PulseSeverity.Normal,
                    Symbol = normalized,
                    Interval = "quote",
                    Success = true,
                    SourceKind = "live_quote_overlay",
                    LatestPrice = quoteResult.Quote.Price,
                    QuoteTimestampUtc = quoteResult.Quote.TimestampUtc
                });

                return new MarketPerceptionResult
                {
                    Success = true,
                    Symbol = normalized,
                    HistorySource = MarketHistorySource.Local,
                    QuoteOverlay = quoteResult.Quote,
                    LiveFetchOccurred = true
                };
            }

            return new MarketPerceptionResult
            {
                Success = false,
                Symbol = normalized,
                HistorySource = MarketHistorySource.Missing,
                ErrorMessage = quoteResult.ErrorMessage ?? "Quote fetch failed.",
                LiveFetchOccurred = true
            };
        }

        public async Task<MarketPerceptionResult> PerceiveCandlesAsync(
            MarketPerceptionRequest request,
            CancellationToken ct = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));

            if (!SymbolValidator.TryNormalize(request.Symbol, out var symbol))
                return new MarketPerceptionResult
                {
                    Success = false,
                    Symbol = request.Symbol ?? "",
                    HistorySource = MarketHistorySource.Missing,
                    ErrorMessage = "Invalid symbol."
                };

            var interval = request.Interval ?? "1d";
            var gateKey = $"{symbol}:{interval}";
            var gate = _fetchGates.GetOrAdd(gateKey, _ => new SemaphoreSlim(1, 1));

            await gate.WaitAsync(ct);
            try
            {
                return await PerceiveCandlesInternalAsync(symbol, interval, request.LookbackDays, ct);
            }
            finally
            {
                gate.Release();
            }
        }

        private async Task<MarketPerceptionResult> PerceiveCandlesInternalAsync(
            string symbol,
            string interval,
            int lookbackDays,
            CancellationToken ct)
        {
            var warnings = new List<string>();

            // ── A) Check local parquet data ──
            var (localAsset, localDataJson, localRowCount) = await ReadLocalHistoryAsync(symbol, interval, lookbackDays, ct);
            var hasLocalData = localAsset is not null
                               && !string.IsNullOrWhiteSpace(localDataJson)
                               && localRowCount > 0;

            var isStale = localAsset?.LastIngestedAtUtc is null
                          || (DateTime.UtcNow - localAsset.LastIngestedAtUtc.Value) > StalenessThreshold;

            // ── B) Local data is fresh — return with optional live quote overlay ──
            if (hasLocalData && !isStale)
            {
                _root._logger.LogDebug("[Perception] {Symbol}/{Interval}: local data fresh ({Rows} rows).", symbol, interval, localRowCount);

                var quoteOverlay = await TryGetQuoteOverlayAsync(symbol, warnings, ct);

                if (quoteOverlay is not null)
                {
                    _ = _root._bus.PublishAsync(new MarketDataEvent
                    {
                        OccurredAtUtc = DateTimeOffset.UtcNow,
                        Source = "Axiom.Market",
                        Kind = "market_perception",
                        Severity = PulseSeverity.Normal,
                        Symbol = symbol,
                        Interval = interval,
                        Success = true,
                        SourceKind = "live_quote_overlay",
                        LatestPrice = quoteOverlay.Price,
                        QuoteTimestampUtc = quoteOverlay.TimestampUtc
                    });
                }

                return new MarketPerceptionResult
                {
                    Success = true,
                    Symbol = symbol,
                    HistorySource = MarketHistorySource.Local,
                    LocalDataJson = localDataJson,
                    LocalRowCount = localRowCount,
                    LocalDataAsOfUtc = localAsset!.LastIngestedAtUtc,
                    ParquetPath = localAsset.ParquetPath,
                    QuoteOverlay = quoteOverlay,
                    LiveFetchOccurred = quoteOverlay is not null,
                    Warnings = warnings
                };
            }

            // ── C) Local data missing or stale — perform on-demand live fetch ──
            var sourceKind = hasLocalData ? "live_refresh" : "live_bootstrap";
            _root._logger.LogInformation(
                "[Perception] {Symbol}/{Interval}: {Source} (local={HasLocal}, stale={IsStale}).",
                symbol, interval, sourceKind, hasLocalData, isStale);

            if (!_root._pythonDispatcher.IsAvailable)
            {
                if (hasLocalData)
                {
                    warnings.Add("Python not available for live refresh; returning stale local data.");
                    return new MarketPerceptionResult
                    {
                        Success = true,
                        Symbol = symbol,
                        HistorySource = MarketHistorySource.Local,
                        LocalDataJson = localDataJson,
                        LocalRowCount = localRowCount,
                        LocalDataAsOfUtc = localAsset?.LastIngestedAtUtc,
                        ParquetPath = localAsset?.ParquetPath,
                        Warnings = warnings
                    };
                }

                return new MarketPerceptionResult
                {
                    Success = false,
                    Symbol = symbol,
                    HistorySource = MarketHistorySource.Missing,
                    ErrorMessage = "Python not available and no local data exists."
                };
            }

            // ── D) Run ingestion for this single symbol ──
            var outRoot = DataLakeRelativePath;
            var ingestionResult = await _root.MarketIngestion.RunIngestionBatchAsync(
                new[] { symbol }, interval, lookbackDays, outRoot, ct);

            if (ingestionResult.Success && ingestionResult.Report is not null)
            {
                // Persist metadata
                await _root.MarketIngestion.ApplyIngestionBatchAsync(
                    new[] { symbol }, interval, ingestionResult, ct);

                // Re-read fresh local data
                var (freshAsset, freshJson, freshRows) = await ReadLocalHistoryAsync(symbol, interval, lookbackDays, ct);
                var quoteOverlay = await TryGetQuoteOverlayAsync(symbol, warnings, ct);

                var successResult = ingestionResult.Report.Results.FirstOrDefault(r =>
                    r.Symbol.Equals(symbol, StringComparison.OrdinalIgnoreCase) && r.IsSuccess);

                // Publish MarketDataEvent
                _ = _root._bus.PublishAsync(new MarketDataEvent
                {
                    OccurredAtUtc = DateTimeOffset.UtcNow,
                    Source = "Axiom.Market",
                    Kind = "market_perception",
                    Severity = PulseSeverity.Normal,
                    Symbol = symbol,
                    Interval = interval,
                    Success = true,
                    SourceKind = sourceKind,
                    RowsWritten = successResult?.RowsWritten,
                    ParquetPath = successResult?.ParquetPath,
                    LatestPrice = quoteOverlay?.Price,
                    QuoteTimestampUtc = quoteOverlay?.TimestampUtc
                });

                return new MarketPerceptionResult
                {
                    Success = true,
                    Symbol = symbol,
                    HistorySource = hasLocalData ? MarketHistorySource.LiveRefresh : MarketHistorySource.LiveBootstrap,
                    LocalDataJson = freshJson,
                    LocalRowCount = freshRows,
                    LocalDataAsOfUtc = freshAsset?.LastIngestedAtUtc ?? DateTime.UtcNow,
                    ParquetPath = successResult?.ParquetPath ?? freshAsset?.ParquetPath,
                    QuoteOverlay = quoteOverlay,
                    LiveFetchOccurred = true,
                    DataPersisted = true,
                    Warnings = warnings
                };
            }

            // ── E) Live fetch failed — fall back to stale local if available ──
            if (hasLocalData)
            {
                warnings.Add($"Live fetch failed: {ingestionResult.ErrorMessage ?? "unknown"}. Returning stale local data.");
                return new MarketPerceptionResult
                {
                    Success = true,
                    Symbol = symbol,
                    HistorySource = MarketHistorySource.Local,
                    LocalDataJson = localDataJson,
                    LocalRowCount = localRowCount,
                    LocalDataAsOfUtc = localAsset?.LastIngestedAtUtc,
                    ParquetPath = localAsset?.ParquetPath,
                    LiveFetchOccurred = true,
                    Warnings = warnings
                };
            }

            return new MarketPerceptionResult
            {
                Success = false,
                Symbol = symbol,
                HistorySource = MarketHistorySource.Missing,
                ErrorMessage = ingestionResult.ErrorMessage ?? "Live fetch failed and no local data available.",
                LiveFetchOccurred = true
            };
        }

        /// <summary>Read local parquet history for a symbol via Python parquet_read.</summary>
        private async Task<(MarketDataAsset? Asset, string? DataJson, int RowCount)> ReadLocalHistoryAsync(
            string symbol, string interval, int days, CancellationToken ct)
        {
            // Check metadata first
            await using var db = await _root._dbFactory.CreateDbContextAsync(ct);
            var asset = await db.MarketDataAssets
                .AsNoTracking()
                .FirstOrDefaultAsync(a => a.Symbol == symbol && a.Interval == interval, ct);

            if (asset is null || string.IsNullOrWhiteSpace(asset.ParquetPath))
                return (null, null, 0);

            if (!_root._pythonDispatcher.IsAvailable)
                return (asset, null, 0);

            // Read parquet via Python
            var absoluteDataRoot = Path.GetFullPath(
                Path.Combine(_root._env.ContentRootPath, DataLakeRelativePath)).Replace("\\", "/");

            var result = await _root._pythonDispatcher.RunParquetReadAsync(
                symbol, Math.Max(days, 5), absoluteDataRoot, ParquetReadTimeoutMs, ct);

            if (!result.Success || string.IsNullOrWhiteSpace(result.Stdout))
                return (asset, null, 0);

            // Extract JSON safely
            var jsonPayload = result.Stdout;
            var jsonStartIndex = jsonPayload.IndexOf('{');
            if (jsonStartIndex < 0)
                return (asset, null, 0);

            jsonPayload = jsonPayload.Substring(jsonStartIndex);

            // Try to extract row count from JSON
            int rowCount = 0;
            try
            {
                using var doc = JsonDocument.Parse(jsonPayload);
                if (doc.RootElement.TryGetProperty("rows", out var rowsProp))
                    rowCount = rowsProp.GetInt32();
                else if (doc.RootElement.TryGetProperty("data", out var dataProp) && dataProp.ValueKind == JsonValueKind.Array)
                    rowCount = dataProp.GetArrayLength();
            }
            catch (JsonException)
            {
                // Non-fatal — we still have the JSON string
            }

            return (asset, jsonPayload, rowCount);
        }

        /// <summary>Try to get a live quote overlay. Non-fatal on failure.</summary>
        private async Task<MarketQuoteDto?> TryGetQuoteOverlayAsync(
            string symbol, List<string> warnings, CancellationToken ct)
        {
            try
            {
                var quoteResult = await GetQuoteAsync(symbol, ct);
                if (quoteResult.Success && quoteResult.Quote is not null)
                    return quoteResult.Quote;

                warnings.Add($"Quote overlay failed: {quoteResult.ErrorMessage ?? "unknown"}");
            }
            catch (Exception ex)
            {
                warnings.Add($"Quote overlay exception: {ex.Message}");
            }

            return null;
        }

        private async Task<(bool Success, string StdoutJson, string ErrorMessage)> RunFetcherAsync(
            string action,
            IReadOnlyList<string> arguments,
            CancellationToken ct)
        {
            if (!_root._pythonDispatcher.IsAvailable)
            {
                return (false, string.Empty, "Python not available. Run setup_venv.ps1 to create the venv.");
            }

            try
            {
                var result = await _root._pythonDispatcher.RunAsync("market", action, arguments, PythonTimeoutMs, ct);

                if (result.TimedOut)
                    return (false, string.Empty, "Python process timed out.");

                if (!result.Success)
                {
                    if (!string.IsNullOrWhiteSpace(result.Stderr))
                        _root._logger.LogWarning("[Market] Python stderr: {Stderr}", result.Stderr);
                    return (false, string.Empty, $"Python exited with code {result.ExitCode}: {result.Stderr}");
                }

                if (string.IsNullOrWhiteSpace(result.Stdout))
                    return (false, string.Empty, "Python returned empty output.");

                return (true, result.Stdout, string.Empty);
            }
            catch (OperationCanceledException)
            {
                return (false, string.Empty, "Request cancelled.");
            }
        }
    }

    private sealed class MarketIngestionGateway : IAxiom.IMarketIngestionGateway
    {
        private const int IngestionTimeoutMs = 90_000;
        private const string WorkerFailureMessage = "Worker process failed or timed out.";

        private readonly Axiom _root;

        public MarketIngestionGateway(Axiom root)
        {
            _root = root;
        }

        public bool IsPythonAvailable => _root._pythonDispatcher.IsAvailable;

        public async Task<IReadOnlyList<string>> GetActiveSymbolsAsync(CancellationToken ct = default)
        {
            await using var db = await _root._dbFactory.CreateDbContextAsync(ct);

            var portfolioSymbols = await db.Positions
                .Where(p => p.IsOpen || p.Quantity > 0)
                .Select(p => p.Symbol)
                .ToListAsync(ct);

            var watchlistSymbols = await db.WatchlistItems
                .Where(w => w.IsActive)
                .Select(w => w.Symbol)
                .ToListAsync(ct);

            return portfolioSymbols
                .Concat(watchlistSymbols)
                .Select(s => s.ToUpperInvariant())
                .Distinct()
                .OrderBy(s => s)
                .ToList();
        }

        public async Task<MarketIngestionRunResult> RunIngestionBatchAsync(
            IReadOnlyList<string> symbols,
            string interval,
            int lookbackDays,
            string outRoot,
            CancellationToken ct = default)
        {
            if (symbols is null)
                throw new ArgumentNullException(nameof(symbols));

            var validSymbols = symbols.Where(SymbolValidator.IsValid).ToList();
            if (validSymbols.Count == 0)
            {
                _root._logger.LogWarning("[Ingestion] No valid symbols in batch. Skipping.");
                return new MarketIngestionRunResult(false, null, WorkerFailureMessage);
            }

            if (validSymbols.Count < symbols.Count)
            {
                var rejected = symbols.Except(validSymbols);
                _root._logger.LogWarning("[Ingestion] Rejected invalid symbols: {Rejected}", string.Join(", ", rejected));
            }

            string symbolsCsv = string.Join(",", validSymbols);

            try
            {
                var result = await _root._pythonDispatcher.RunMarketIngestAsync(
                    symbolsCsv, interval, lookbackDays, outRoot, IngestionTimeoutMs, ct);

                if (result.TimedOut)
                {
                    _root._logger.LogWarning("[Ingestion] Worker timed out after {Seconds}s for batch: {Symbols}",
                        IngestionTimeoutMs / 1000, symbolsCsv);
                    return new MarketIngestionRunResult(false, null, WorkerFailureMessage);
                }

                if (!string.IsNullOrWhiteSpace(result.Stderr))
                {
                    _root._logger.LogDebug("[Ingestion] Worker stderr:\n{Stderr}", result.Stderr);
                }

                if (!result.Success)
                {
                    if (ct.IsCancellationRequested)
                    {
                        _root._logger.LogInformation("[Ingestion] Worker cancelled (app shutting down).");
                        return new MarketIngestionRunResult(false, null, WorkerFailureMessage);
                    }

                    _root._logger.LogWarning("[Ingestion] Worker exited with code {ExitCode} for batch: {Symbols}",
                        result.ExitCode, symbolsCsv);
                    return new MarketIngestionRunResult(false, null, WorkerFailureMessage);
                }

                if (string.IsNullOrWhiteSpace(result.Stdout))
                {
                    _root._logger.LogWarning("[Ingestion] Worker returned empty stdout.");
                    return new MarketIngestionRunResult(false, null, WorkerFailureMessage);
                }

                try
                {
                    var report = JsonSerializer.Deserialize<IngestionReport>(result.Stdout.Trim());
                    if (report is null)
                    {
                        return new MarketIngestionRunResult(false, null, WorkerFailureMessage);
                    }

                    return new MarketIngestionRunResult(true, report, null);
                }
                catch (JsonException ex)
                {
                    _root._logger.LogError(ex, "[Ingestion] Failed to parse ingestion report JSON.");
                    return new MarketIngestionRunResult(false, null, WorkerFailureMessage);
                }
            }
            catch (OperationCanceledException)
            {
                _root._logger.LogInformation("[Ingestion] Worker run cancelled.");
                return new MarketIngestionRunResult(false, null, WorkerFailureMessage);
            }
            catch (Exception ex)
            {
                _root._logger.LogError(ex, "[Ingestion] Process error.");
                return new MarketIngestionRunResult(false, null, WorkerFailureMessage);
            }
        }

        public async Task ApplyIngestionBatchAsync(
            IReadOnlyList<string> batchSymbols,
            string interval,
            MarketIngestionRunResult runResult,
            CancellationToken ct = default)
        {
            if (batchSymbols is null)
                throw new ArgumentNullException(nameof(batchSymbols));

            await using var db = await _root._dbFactory.CreateDbContextAsync(ct);

            if (!runResult.Success || runResult.Report is null)
            {
                foreach (var symbol in batchSymbols)
                {
                    await UpsertAssetFailureAsync(
                        db,
                        symbol,
                        interval,
                        runResult.ErrorMessage ?? WorkerFailureMessage,
                        ct);
                }

                await db.SaveChangesAsync(ct);
                return;
            }

            var report = runResult.Report;

            foreach (var result in report.Results)
            {
                var asset = await db.MarketDataAssets
                    .FirstOrDefaultAsync(
                        a => a.Symbol == result.Symbol && a.Interval == result.Interval,
                        ct);

                if (asset is null)
                {
                    asset = new MarketDataAsset
                    {
                        Symbol = result.Symbol,
                        Interval = result.Interval
                    };
                    db.MarketDataAssets.Add(asset);
                }

                if (result.IsSuccess)
                {
                    asset.ParquetPath = result.ParquetPath;
                    asset.LastIngestedAtUtc = DateTime.UtcNow;
                    asset.ProviderUsed = result.ProviderUsed;
                    asset.RowsWritten = result.RowsWritten;
                    asset.ConsecutiveFailures = 0;
                    asset.LastError = null;

                    if (DateTime.TryParse(result.DataEndUtc, CultureInfo.InvariantCulture,
                            DateTimeStyles.RoundtripKind, out var dataEnd))
                    {
                        asset.LastDataEndUtc = dataEnd;
                    }

                    _root._logger.LogInformation("[Ingestion]   {Symbol}: OK via {Provider} ({Rows} rows -> {Path})",
                        result.Symbol, result.ProviderUsed, result.RowsWritten, result.ParquetPath);
                }
                else
                {
                    asset.ConsecutiveFailures++;
                    asset.LastError = result.Error?.Message ?? "Unknown error";
                    _root._logger.LogWarning("[Ingestion]   {Symbol}: FAIL ({Failures}x) - {Error}",
                        result.Symbol, asset.ConsecutiveFailures, asset.LastError);
                }

                asset.UpdatedAtUtc = DateTime.UtcNow;
            }

            await db.SaveChangesAsync(ct);

            if (report.Warnings.Count > 0)
            {
                foreach (var warning in report.Warnings)
                {
                    _root._logger.LogWarning("[Ingestion] Worker warning: {Warning}", warning);
                }
            }
        }

        private static async Task UpsertAssetFailureAsync(
            AppDbContext db,
            string symbol,
            string interval,
            string error,
            CancellationToken ct)
        {
            var asset = await db.MarketDataAssets
                .FirstOrDefaultAsync(a => a.Symbol == symbol && a.Interval == interval, ct);

            if (asset is null)
            {
                asset = new MarketDataAsset { Symbol = symbol, Interval = interval };
                db.MarketDataAssets.Add(asset);
            }

            asset.ConsecutiveFailures++;
            asset.LastError = error;
            asset.UpdatedAtUtc = DateTime.UtcNow;
        }
    }

    private sealed class McpGateway : IAxiom.IMcpGateway
    {
        private readonly Axiom _root;

        public McpGateway(Axiom root)
        {
            _root = root;
        }

        public IReadOnlyList<System.Text.Json.Nodes.JsonNode> GetOpenAiToolSchemas()
        {
            return _root._mcpSchemaAdapter.GetOpenAiToolSchemas();
        }

        public bool IsStateChangingTool(string toolName)
        {
            return _root._mcpInvoker.IsStateChangingTool(toolName);
        }

        public async Task<McpInvokeResult> InvokeAsync(
            string toolName,
            string argumentsJson,
            CancellationToken ct = default)
        {
            var result = await _root._mcpInvoker.InvokeAsync(toolName, argumentsJson, ct);
            return new McpInvokeResult(
                result.ToolContent,
                result.UiActions.ToArray(),
                result.IsSuccess,
                result.Error);
        }
    }

    private sealed class TradeGateway : IAxiom.ITradeGateway
    {
        private readonly Axiom _root;

        public TradeGateway(Axiom root)
        {
            _root = root;
        }

        public async Task<TradeExecutionResult> ExecuteTradeAsync(
            ExecuteTradeRequest request,
            CancellationToken ct = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));

            if (string.IsNullOrWhiteSpace(request.ClientRequestId))
            {
                _root._logger.LogWarning("[Axiom] Trade rejected: missing ClientRequestId.");
                return new TradeExecutionResult(false, false, false, null, "ClientRequestId is required.");
            }

            if (!SymbolValidator.TryNormalize(request.Symbol, out var normalizedSymbol))
            {
                _root._logger.LogWarning("[Axiom] Trade rejected: invalid symbol '{Symbol}'.", request.Symbol);
                return new TradeExecutionResult(false, false, false, null, "Invalid symbol format.");
            }

            string side = (request.Side ?? string.Empty).Trim().ToUpperInvariant();
            if (side is not ("BUY" or "SELL"))
            {
                return new TradeExecutionResult(false, false, false, null, "Side must be BUY or SELL.");
            }

            if (request.Quantity <= 0)
            {
                return new TradeExecutionResult(false, false, false, null, "Quantity must be greater than 0.");
            }

            if (request.ExecutedPrice <= 0)
            {
                return new TradeExecutionResult(false, false, false, null, "ExecutedPrice must be greater than 0.");
            }

            ct.ThrowIfCancellationRequested();

            using var scope = _root._scopeFactory.CreateScope();
            var tradingService = scope.ServiceProvider.GetRequiredService<TradingService>();

            var tradeRequest = new TradeRequest
            {
                ClientRequestId = request.ClientRequestId,
                Symbol = normalizedSymbol,
                Side = side,
                Quantity = request.Quantity,
                ExecutedPrice = request.ExecutedPrice,
                Fees = request.Fees,
                Currency = request.Currency,
                Notes = request.Notes,
                RawJson = request.RawJson
            };

            var result = await tradingService.ExecuteTradeAsync(tradeRequest);
            return new TradeExecutionResult(
                result.Success,
                result.IsDuplicate,
                result.IsConflict,
                result.Trade is null ? null : MapTrade(result.Trade),
                result.ErrorMessage);
        }
    }

    private sealed class ChatGateway : IAxiom.IChatGateway
    {
        private static readonly HashSet<string> AllowedRoles = new(StringComparer.OrdinalIgnoreCase)
        {
            "system", "user", "assistant", "tool"
        };

        private readonly Axiom _root;

        public ChatGateway(Axiom root)
        {
            _root = root;
        }

        public async Task<IReadOnlyList<ChatMessageDto>> GetThreadMessagesAsync(
            string threadId,
            CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(threadId))
                throw new ArgumentException("ThreadId is required.", nameof(threadId));

            await using var db = await _root._dbFactory.CreateDbContextAsync(ct);
            return await db.ChatMessages
                .AsNoTracking()
                .Where(m => m.ThreadId == threadId)
                .OrderBy(m => m.CreatedAtUtc)
                .Select(m => new ChatMessageDto(
                    m.ThreadId,
                    m.Role,
                    m.Content,
                    m.MetadataJson,
                    m.CreatedAtUtc))
                .ToListAsync(ct);
        }

        public async Task AppendMessageAsync(
            ChatMessageWrite message,
            CancellationToken ct = default)
        {
            if (message is null)
                throw new ArgumentNullException(nameof(message));

            if (string.IsNullOrWhiteSpace(message.ThreadId))
                throw new ArgumentException("ThreadId is required.", nameof(message));

            if (!AllowedRoles.Contains(message.Role))
                throw new ArgumentException($"Unsupported chat role: '{message.Role}'.", nameof(message));

            await using var db = await _root._dbFactory.CreateDbContextAsync(ct);
            db.ChatMessages.Add(new ChatMessage
            {
                ThreadId = message.ThreadId,
                Role = message.Role,
                Content = message.Content ?? string.Empty,
                MetadataJson = message.MetadataJson,
                CreatedAtUtc = message.CreatedAtUtc ?? DateTime.UtcNow
            });

            await db.SaveChangesAsync(ct);
        }
    }

    private sealed class ToolRunGateway : IAxiom.IToolRunGateway
    {
        private readonly Axiom _root;

        public ToolRunGateway(Axiom root)
        {
            _root = root;
        }

        public async Task AppendBatchAsync(
            IReadOnlyList<ToolRunWrite> runs,
            CancellationToken ct = default)
        {
            if (runs is null)
                throw new ArgumentNullException(nameof(runs));

            if (runs.Count == 0)
                return;

            await using var db = await _root._dbFactory.CreateDbContextAsync(ct);
            foreach (var run in runs)
            {
                db.ToolRuns.Add(new ToolRun
                {
                    ThreadId = run.ThreadId,
                    ToolName = run.ToolName,
                    ArgumentsJson = run.ArgumentsJson,
                    ResultJson = run.ResultJson,
                    ExecutionTimeMs = run.ExecutionTimeMs,
                    IsSuccess = run.IsSuccess,
                    CreatedAtUtc = run.CreatedAtUtc
                });
            }

            await db.SaveChangesAsync(ct);
        }
    }

    private sealed class SkillGateway : IAxiom.ISkillGateway
    {
        private static readonly JsonSerializerOptions JsonOpts = new()
        {
            WriteIndented = false
        };

        private readonly Axiom _root;

        public SkillGateway(Axiom root)
        {
            _root = root;
        }

        public string GetAvailableSkills(bool includeDeprecated = false)
        {
            var snapshot = _root._skillRegistry.Snapshot;
            var skills = snapshot.Playbooks
                .Where(p => includeDeprecated || !p.Metadata.Deprecated)
                .Select(p => new Dictionary<string, object?>
                {
                    ["skill_name"] = p.Metadata.SkillName,
                    ["display_name"] = p.Metadata.DisplayName,
                    ["version"] = p.Metadata.Version,
                    ["description"] = p.Metadata.Description,
                    ["tags"] = p.Metadata.Tags,
                    ["required_tools"] = p.Metadata.RequiredTools,
                    ["deprecated"] = p.Metadata.Deprecated
                })
                .ToList();

            var response = new Dictionary<string, object?>
            {
                ["ok"] = true,
                ["schema_version"] = 1,
                ["skills"] = skills,
                ["total_count"] = skills.Count
            };

            return JsonSerializer.Serialize(response, JsonOpts);
        }

        public string ReadPlaybook(string skillName)
        {
            if (string.IsNullOrWhiteSpace(skillName) || !SkillNamePattern.IsMatch(skillName))
            {
                return BuildErrorResponse(
                    "invalid_skill_name",
                    $"Invalid skill_name: '{skillName}'. Must match pattern: ^[a-z][a-z0-9_]{{0,63}}$");
            }

            if (skillName.Contains('/') || skillName.Contains('\\') || skillName.Contains(".."))
            {
                return BuildErrorResponse("invalid_skill_name", "Skill name must not contain path characters.");
            }

            var snapshot = _root._skillRegistry.Snapshot;
            if (!snapshot.BySkillName.TryGetValue(skillName, out var playbook))
            {
                return BuildErrorResponse("skill_not_found", $"Skill '{skillName}' not found.");
            }

            var metadata = new Dictionary<string, object?>
            {
                ["skill_name"] = playbook.Metadata.SkillName,
                ["display_name"] = playbook.Metadata.DisplayName,
                ["version"] = playbook.Metadata.Version,
                ["description"] = playbook.Metadata.Description,
                ["tags"] = playbook.Metadata.Tags,
                ["required_tools"] = playbook.Metadata.RequiredTools,
                ["deprecated"] = playbook.Metadata.Deprecated,
                ["extras"] = playbook.Metadata.Extras.Count > 0
                    ? playbook.Metadata.Extras
                    : new Dictionary<string, object>()
            };

            var response = new Dictionary<string, object?>
            {
                ["ok"] = true,
                ["skill_name"] = playbook.Metadata.SkillName,
                ["metadata"] = metadata,
                ["markdown_body"] = playbook.MarkdownBody,
                ["content_hash"] = playbook.ContentHash,
                ["file_size_bytes"] = playbook.FileSizeBytes
            };

            return JsonSerializer.Serialize(response, JsonOpts);
        }

        private static string BuildErrorResponse(string error, string message)
        {
            var response = new Dictionary<string, object?>
            {
                ["ok"] = false,
                ["error"] = error,
                ["message"] = message
            };

            return JsonSerializer.Serialize(response, JsonOpts);
        }
    }

    private sealed class PerceptionGateway : IAxiom.IPerceptionGateway
    {
        private const int IngestTimeoutMs = 120_000; // 2 min — fetches 7 proxies + calendar + headlines
        private const int SnapshotTimeoutMs = 15_000; // local reads only

        private readonly Axiom _root;

        public PerceptionGateway(Axiom root) => _root = root;

        public async Task<PerceptionIngestResult> RunIngestAsync(
            int lookbackDays = 365,
            int headlineLimit = 15,
            int calendarHorizonDays = 30,
            CancellationToken ct = default)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            if (!_root._pythonDispatcher.IsAvailable)
            {
                return new PerceptionIngestResult(
                    false, 0, 0, false, null, 0, false, null, 0, null, 0,
                    "Python not available");
            }

            var result = await _root._pythonDispatcher.RunPerceptionIngestAsync(
                lookbackDays, headlineLimit, calendarHorizonDays, IngestTimeoutMs, ct);

            sw.Stop();

            if (!result.Success || result.TimedOut)
            {
                return new PerceptionIngestResult(
                    false, 0, 0, false, null, 0, false, null, 0, null, sw.ElapsedMilliseconds,
                    result.TimedOut ? "Perception ingest timed out" : result.Stderr);
            }

            try
            {
                using var doc = JsonDocument.Parse(result.Stdout);
                var root = doc.RootElement;

                var proxies = root.GetProperty("proxies");
                var calendar = root.GetProperty("calendar");
                var headlines = root.GetProperty("headlines");

                return new PerceptionIngestResult(
                    Success: root.GetProperty("ok").GetBoolean(),
                    ProxiesSucceeded: proxies.GetProperty("succeeded").GetInt32(),
                    ProxiesTotal: proxies.GetProperty("total").GetInt32(),
                    CalendarOk: calendar.GetProperty("ok").GetBoolean(),
                    CalendarProvider: calendar.TryGetProperty("provider", out var cp) ? cp.GetString() : null,
                    CalendarEventCount: calendar.GetProperty("eventCount").GetInt32(),
                    HeadlinesOk: headlines.GetProperty("ok").GetBoolean(),
                    HeadlinesProvider: headlines.TryGetProperty("provider", out var hp) ? hp.GetString() : null,
                    HeadlineCount: headlines.GetProperty("headlineCount").GetInt32(),
                    ManifestPath: root.TryGetProperty("manifestPath", out var mp) ? mp.GetString() : null,
                    DurationMs: sw.ElapsedMilliseconds,
                    ErrorMessage: null);
            }
            catch (Exception ex)
            {
                return new PerceptionIngestResult(
                    false, 0, 0, false, null, 0, false, null, 0, null, sw.ElapsedMilliseconds,
                    $"Failed to parse ingest report: {ex.Message}");
            }
        }

        public async Task<PythonRouteResult> ReadSnapshotAsync(
            int headlineLimit = 10,
            CancellationToken ct = default)
        {
            if (!_root._pythonDispatcher.IsAvailable)
            {
                return new PythonRouteResult(false, "", "Python not available", -1, false);
            }

            var result = await _root._pythonDispatcher.RunPerceptionSnapshotAsync(
                headlineLimit, SnapshotTimeoutMs, ct);

            return new PythonRouteResult(
                result.Success && !result.TimedOut,
                result.Stdout,
                result.Stderr,
                result.ExitCode,
                result.TimedOut);
        }
    }

    private static TradeSnapshotDto MapTrade(Trade trade)
    {
        return new TradeSnapshotDto(
            trade.Id,
            trade.ClientRequestId,
            trade.Symbol,
            trade.Side,
            trade.Quantity,
            trade.ExecutedPrice,
            trade.Fees,
            trade.Currency,
            trade.ExecutedAtUtc,
            trade.Status,
            trade.Notes,
            trade.RawJson,
            trade.PositionId);
    }
}
