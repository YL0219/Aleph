using System.ComponentModel;
using System.Text.Json;
using Microsoft.Extensions.Hosting;
using ModelContextProtocol.Server;
using Microsoft.Extensions.Logging;

namespace Aleph
{
    [McpServerToolType]
    public class McpMarketTools
    {
        private const int MinDays = 5;
        private const int MaxDays = 365;
        private const int ParquetReadTimeoutMs = 30_000;

        private static readonly JsonSerializerOptions JsonOpts = new()
        {
            WriteIndented = false
        };

        private readonly PythonDispatcherService _dispatcher;
        private readonly IAxiom _axiom;
        private readonly ILogger<McpMarketTools> _logger;
        private readonly IHostEnvironment _env;

        public McpMarketTools(
            PythonDispatcherService dispatcher,
            IAxiom axiom,
            ILogger<McpMarketTools> logger,
            IHostEnvironment env)
        {
            _dispatcher = dispatcher;
            _axiom = axiom;
            _logger = logger;
            _env = env;
        }

        [McpServerTool(Name = "query_local_market_data", ReadOnly = true)]
        [Description(
            "Query locally-stored OHLCV market data for a market symbol. " +
            "Returns daily candles (open, high, low, close, volume) from the Parquet data lake. " +
            "Data is sourced from prior ingestion cycles — not a live API call. " +
            "Use this for historical analysis, trend review, and portfolio monitoring.")]
        public async Task<string> QueryLocalMarketData(
            [Description("Market symbol (e.g. SI=F, GC=F, AAPL, BRK.B). 1-15 uppercase characters using A-Z, 0-9, dot, hyphen, equals.")]
            string symbol,
            [Description("Number of days of historical data to retrieve. MINIMUM IS 5 (to account for weekends). Defaults to 7.")]
            int days = 7)
        {
            if (!SymbolValidator.TryNormalize(symbol, out var normalizedSymbol))
            {
                return BuildErrorJson($"Invalid symbol: '{symbol}'.");
            }

            var clampedDays = Math.Clamp(days, MinDays, MaxDays);

            if (!_dispatcher.IsAvailable)
            {
                return BuildErrorJson("Python environment not available.");
            }

            // FIX 1: Force forward slashes so Python doesn't break on Windows paths
            var absoluteDataRoot = Path.GetFullPath(Path.Combine(_env.ContentRootPath, "data_lake", "market", "ohlcv")).Replace("\\", "/");

            _logger.LogInformation("[MCP] Executing Parquet Read. Symbol={Symbol}, Root={Root}", normalizedSymbol, absoluteDataRoot);

            var result = await _dispatcher.RunParquetReadAsync(
                normalizedSymbol,
                clampedDays,
                absoluteDataRoot,
                ParquetReadTimeoutMs);

            // --- THE X-RAY LOGS ---
            _logger.LogInformation("[MCP] Python ExitCode: {Code}", result.ExitCode);
            _logger.LogInformation("[MCP] Python Stdout: {Stdout}", result.Stdout);
            
            if (!string.IsNullOrWhiteSpace(result.Stderr)) 
            {
                _logger.LogWarning("[MCP] Python Stderr: {Stderr}", result.Stderr);
            }

            // FIX 2: Safely extract JSON even if Python printed weird warnings before it
            var jsonPayload = result.Stdout ?? string.Empty;
            var jsonStartIndex = jsonPayload.IndexOf('{');
            if (jsonStartIndex >= 0)
            {
                // Cut out any warnings that appeared before the JSON
                jsonPayload = jsonPayload.Substring(jsonStartIndex);
                return jsonPayload;
            }

            if (!result.Success)
            {
                var reason = result.TimedOut
                    ? "Parquet read timed out."
                    : $"Parquet read failed (exit code {result.ExitCode}). {result.Stderr}";

                return BuildErrorJson(reason);
            }

            return result.Stdout ?? BuildErrorJson("Empty output from Python.");
        }

        [McpServerTool(Name = "perceive_market_data", ReadOnly = true)]
        [Description(
            "Perceive market data for ANY market symbol — even if it's not on the background watchlist. " +
            "Uses a hybrid local-first + live-fetch model: returns cached Parquet history if available, " +
            "otherwise fetches live data and persists it. Also provides a live quote overlay when possible. " +
            "Use this when you need data for a symbol the system hasn't tracked before.")]
        public async Task<string> PerceiveMarketData(
            [Description("Market symbol (e.g. SI=F, CL=F, GLD, AAPL). 1-15 uppercase characters using A-Z, 0-9, dot, hyphen, equals.")]
            string symbol,
            [Description("Number of days of historical data. MINIMUM IS 5. Defaults to 90.")]
            int days = 90,
            [Description("Candle interval. Defaults to '1d'.")]
            string interval = "1d")
        {
            if (!SymbolValidator.TryNormalize(symbol, out var normalizedSymbol))
            {
                return BuildErrorJson($"Invalid symbol: '{symbol}'.");
            }

            var clampedDays = Math.Clamp(days, MinDays, MaxDays);

            _logger.LogInformation("[MCP] Perceive market data: {Symbol}, {Days}d, {Interval}", normalizedSymbol, clampedDays, interval);

            var result = await _axiom.Market.PerceiveCandlesAsync(
                new MarketPerceptionRequest
                {
                    Symbol = normalizedSymbol,
                    Interval = interval,
                    LookbackDays = clampedDays
                });

            var response = new Dictionary<string, object?>
            {
                ["ok"] = result.Success,
                ["symbol"] = result.Symbol,
                ["historySource"] = result.HistorySource.ToString().ToLowerInvariant(),
                ["liveFetchOccurred"] = result.LiveFetchOccurred,
                ["dataPersisted"] = result.DataPersisted,
                ["localRowCount"] = result.LocalRowCount,
                ["localDataAsOfUtc"] = result.LocalDataAsOfUtc?.ToString("o"),
                ["parquetPath"] = result.ParquetPath
            };

            if (result.QuoteOverlay is not null)
            {
                response["quoteOverlay"] = new Dictionary<string, object?>
                {
                    ["price"] = result.QuoteOverlay.Price,
                    ["timestampUtc"] = result.QuoteOverlay.TimestampUtc
                };
            }

            if (result.Warnings.Count > 0)
                response["warnings"] = result.Warnings;

            if (!result.Success)
                response["error"] = result.ErrorMessage;

            // If we have local data JSON, parse and embed it
            if (!string.IsNullOrWhiteSpace(result.LocalDataJson))
            {
                try
                {
                    using var doc = JsonDocument.Parse(result.LocalDataJson);
                    response["marketData"] = doc.RootElement.Clone();
                }
                catch (JsonException)
                {
                    response["marketData"] = result.LocalDataJson;
                }
            }

            return JsonSerializer.Serialize(response, JsonOpts);
        }

        private static string BuildErrorJson(string message)
        {
            var escaped = message.Replace("\\", "\\\\").Replace("\"", "\\\"");
            return $"{{\"ok\":false,\"error\":\"{escaped}\"}}";
        }
    }
}
