using System.ComponentModel;
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

        private readonly PythonDispatcherService _dispatcher;
        private readonly ILogger<McpMarketTools> _logger;
        private readonly IHostEnvironment _env;

        public McpMarketTools(
            PythonDispatcherService dispatcher,
            ILogger<McpMarketTools> logger,
            IHostEnvironment env)
        {
            _dispatcher = dispatcher;
            _logger = logger;
            _env = env;
        }

        [McpServerTool(Name = "query_local_market_data", ReadOnly = true)]
        [Description(
            "Query locally-stored OHLCV market data for a stock symbol. " +
            "Returns daily candles (open, high, low, close, volume) from the Parquet data lake. " +
            "Data is sourced from prior ingestion cycles — not a live API call. " +
            "Use this for historical analysis, trend review, and portfolio monitoring.")]
        public async Task<string> QueryLocalMarketData(
            [Description("Stock ticker symbol (e.g. AAPL, MSFT, BRK.B). 1-15 uppercase alphanumeric characters.")]
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

        private static string BuildErrorJson(string message)
        {
            var escaped = message.Replace("\\", "\\\\").Replace("\"", "\\\"");
            return $"{{\"ok\":false,\"error\":\"{escaped}\"}}";
        }
    }
}