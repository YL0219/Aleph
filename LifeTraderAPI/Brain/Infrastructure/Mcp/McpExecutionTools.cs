using System.ComponentModel;
using LifeTrader_AI.Services;
using ModelContextProtocol.Server;

namespace LifeTrader_AI.Infrastructure.Mcp
{
    /// <summary>
    /// MCP tools for trade execution and UI actions.
    /// </summary>
    [McpServerToolType]
    public sealed class McpExecutionTools
    {
        private const int MaxShares = 1_000_000;
        private const decimal MaxPrice = 1_000_000m;

        private static readonly HashSet<string> AllowedTimeframes = new(StringComparer.OrdinalIgnoreCase)
        {
            "1m", "5m", "15m", "1h", "1d", "1w", "1mo"
        };

        private static readonly HashSet<string> AllowedRanges = new(StringComparer.OrdinalIgnoreCase)
        {
            "7d", "30d", "90d", "180d", "1y", "2y"
        };

        private readonly IServiceScopeFactory _scopeFactory;
        private readonly ILogger<McpExecutionTools> _logger;

        public McpExecutionTools(
            IServiceScopeFactory scopeFactory,
            ILogger<McpExecutionTools> logger)
        {
            _scopeFactory = scopeFactory;
            _logger = logger;
        }

        [McpServerTool(Name = "execute_trade", ReadOnly = false)]
        [Description("Buys or sells shares of a stock and updates the user's portfolio.")]
        public async Task<string> ExecuteTrade(
            [Description("Trade action. Must be 'buy' or 'sell'.")]
            string action,
            [Description("Stock ticker symbol, e.g., AAPL.")]
            string symbol,
            [Description("Number of shares to trade (must be greater than 0).")]
            int shares,
            [Description("Executed price per share in USD (must be greater than 0).")]
            decimal price,
            CancellationToken ct = default)
        {
            var result = await ExecuteTradeInternalAsync(action, symbol, shares, price, ct);
            return result.ToolContent;
        }

        [McpServerTool(Name = "open_chart", ReadOnly = true)]
        [Description("Opens an interactive candlestick chart for a symbol.")]
        public Task<string> OpenChart(
            [Description("Stock ticker symbol, e.g., AAPL.")]
            string symbol,
            [Description("Timeframe: 1m, 5m, 15m, 1h, 1d, 1w, or 1mo. Defaults to 1d.")]
            string tf = "1d",
            [Description("Date range: 7d, 30d, 90d, 180d, 1y, or 2y. Defaults to 180d.")]
            string range = "180d",
            CancellationToken ct = default)
        {
            var result = OpenChartInternal(symbol, tf, range);
            return Task.FromResult(result.ToolContent);
        }

        internal async Task<McpToolResult> ExecuteTradeInternalAsync(
            string action,
            string symbol,
            int shares,
            decimal price,
            CancellationToken ct = default)
        {
            string normalizedAction = (action ?? "").Trim().ToLowerInvariant();
            if (normalizedAction is not ("buy" or "sell"))
                return McpToolResult.Failure("ERROR: Invalid action. Must be 'buy' or 'sell'.");

            if (!SymbolValidator.TryNormalize(symbol, out var normalizedSymbol))
                return McpToolResult.Failure("ERROR: Invalid symbol format.");

            if (shares <= 0)
                return McpToolResult.Failure("ERROR: Shares must be greater than 0.");

            if (price <= 0m)
                return McpToolResult.Failure("ERROR: Price must be greater than 0.");

            int clampedShares = Math.Min(shares, MaxShares);
            decimal clampedPrice = Math.Min(price, MaxPrice);

            if (clampedShares != shares)
                _logger.LogWarning("[MCP] execute_trade shares clamped from {Requested} to {Clamped}",
                    shares, clampedShares);

            if (clampedPrice != price)
                _logger.LogWarning("[MCP] execute_trade price clamped from {Requested} to {Clamped}",
                    price, clampedPrice);

            using var scope = _scopeFactory.CreateScope();
            var tradingService = scope.ServiceProvider.GetRequiredService<TradingService>();

            var tradeReq = new TradeRequest
            {
                ClientRequestId = Guid.NewGuid().ToString(),
                Symbol = normalizedSymbol,
                Side = normalizedAction.ToUpperInvariant(),
                Quantity = clampedShares,
                ExecutedPrice = clampedPrice,
                Currency = "USD"
            };

            var result = await tradingService.ExecuteTradeAsync(tradeReq);
            if (!result.Success)
                return McpToolResult.Failure($"ERROR: {result.ErrorMessage}");

            return McpToolResult.Success(
                $"SUCCESS: {normalizedAction.ToUpperInvariant()} {clampedShares} shares of {normalizedSymbol} at ${clampedPrice}.");
        }

        internal McpToolResult OpenChartInternal(string symbol, string? tf, string? range)
        {
            if (!SymbolValidator.TryNormalize(symbol, out var normalizedSymbol))
                return McpToolResult.Failure("ERROR: Invalid symbol format.");

            string normalizedTf = NormalizeTf(tf);
            string normalizedRange = NormalizeRange(range);

            string chartUrl = $"/chart/index.html?symbol={normalizedSymbol}&tf={normalizedTf}&range={normalizedRange}";
            var uiAction = new
            {
                type = "openChart",
                symbol = normalizedSymbol,
                tf = normalizedTf,
                range = normalizedRange,
                url = chartUrl
            };

            _logger.LogInformation("[MCP] UI action: openChart -> {ChartUrl}", chartUrl);

            return McpToolResult.Success(
                $"Chart opened for {normalizedSymbol} tf={normalizedTf} range={normalizedRange}",
                new[] { (object)uiAction });
        }

        private static string NormalizeTf(string? tf)
        {
            string candidate = string.IsNullOrWhiteSpace(tf) ? "1d" : tf.Trim();
            return AllowedTimeframes.Contains(candidate) ? candidate : "1d";
        }

        private static string NormalizeRange(string? range)
        {
            string candidate = string.IsNullOrWhiteSpace(range) ? "180d" : range.Trim();
            return AllowedRanges.Contains(candidate) ? candidate : "180d";
        }
    }
}
