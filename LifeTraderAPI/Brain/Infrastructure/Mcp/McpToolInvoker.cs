using System.Globalization;
using System.Text.Json;

namespace LifeTrader_AI.Infrastructure.Mcp
{
    /// <summary>
    /// Single execution gateway for all MCP-backed AI tool calls.
    /// </summary>
    public sealed class McpToolInvoker
    {
        private sealed record ToolRoute(
            bool IsStateChanging,
            Func<JsonElement, CancellationToken, Task<McpToolResult>> Handler);

        private readonly McpMarketTools _marketTools;
        private readonly McpExecutionTools _executionTools;
        private readonly McpNewsTools _newsTools;
        private readonly ILogger<McpToolInvoker> _logger;
        private readonly IReadOnlyDictionary<string, ToolRoute> _routes;
        private readonly IReadOnlySet<string> _stateChangingTools;

        public McpToolInvoker(
            McpMarketTools marketTools,
            McpExecutionTools executionTools,
            McpNewsTools newsTools,
            ILogger<McpToolInvoker> logger)
        {
            _marketTools = marketTools;
            _executionTools = executionTools;
            _newsTools = newsTools;
            _logger = logger;

            var routes = new Dictionary<string, ToolRoute>(StringComparer.OrdinalIgnoreCase)
            {
                ["query_local_market_data"] = new(
                    IsStateChanging: false,
                    Handler: InvokeQueryLocalMarketDataAsync),
                ["execute_trade"] = new(
                    IsStateChanging: true,
                    Handler: InvokeExecuteTradeAsync),
                ["open_chart"] = new(
                    IsStateChanging: false,
                    Handler: InvokeOpenChartAsync),
                ["get_news_headlines"] = new(
                    IsStateChanging: false,
                    Handler: InvokeGetNewsHeadlinesAsync),
                ["scrape_website_text"] = new(
                    IsStateChanging: false,
                    Handler: InvokeScrapeWebsiteTextAsync)
            };

            _routes = routes;
            _stateChangingTools = routes
                .Where(kv => kv.Value.IsStateChanging)
                .Select(kv => kv.Key)
                .ToHashSet(StringComparer.OrdinalIgnoreCase);
        }

        public bool IsStateChangingTool(string toolName) => _stateChangingTools.Contains(toolName);

        public async Task<McpToolResult> InvokeAsync(
            string toolName,
            string argumentsJson,
            CancellationToken ct = default)
        {
            _logger.LogDebug("[McpInvoker] Invoking tool: {ToolName}", toolName);

            if (!_routes.TryGetValue(toolName, out var route))
            {
                string msg = $"Unknown MCP tool: '{toolName}'.";
                _logger.LogWarning("[McpInvoker] {Message}", msg);
                return BuildInvokerFailure(msg);
            }

            try
            {
                string normalizedJson = string.IsNullOrWhiteSpace(argumentsJson) ? "{}" : argumentsJson;
                using var doc = JsonDocument.Parse(normalizedJson);
                return await route.Handler(doc.RootElement, ct);
            }
            catch (JsonException ex)
            {
                _logger.LogWarning(ex, "[McpInvoker] Invalid args JSON for tool '{ToolName}'", toolName);
                return BuildInvokerFailure($"Invalid arguments JSON for tool '{toolName}'.");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[McpInvoker] Unhandled exception invoking tool '{ToolName}'", toolName);
                return BuildInvokerFailure($"Internal error invoking tool '{toolName}'.");
            }
        }

        private async Task<McpToolResult> InvokeQueryLocalMarketDataAsync(
            JsonElement root,
            CancellationToken ct)
        {
            if (!TryGetRequiredString(root, "symbol", out string symbol, out string err))
                return BuildInvokerFailure(err);

            int days = 7;
            if (root.TryGetProperty("days", out var daysProp))
            {
                if (!TryReadInt(daysProp, out days))
                    return BuildInvokerFailure("Argument 'days' must be an integer.");
            }

            string content = await _marketTools.QueryLocalMarketData(symbol, days);
            return InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content);
        }

        private Task<McpToolResult> InvokeExecuteTradeAsync(JsonElement root, CancellationToken ct)
        {
            if (!TryGetRequiredString(root, "action", out string action, out string actionErr))
                return Task.FromResult(BuildInvokerFailure(actionErr));

            if (!TryGetRequiredString(root, "symbol", out string symbol, out string symbolErr))
                return Task.FromResult(BuildInvokerFailure(symbolErr));

            if (!TryGetRequiredInt(root, "shares", out int shares, out string sharesErr))
                return Task.FromResult(BuildInvokerFailure(sharesErr));

            if (!TryGetRequiredDecimal(root, "price", out decimal price, out string priceErr))
                return Task.FromResult(BuildInvokerFailure(priceErr));

            return _executionTools.ExecuteTradeInternalAsync(action, symbol, shares, price, ct);
        }

        private Task<McpToolResult> InvokeOpenChartAsync(JsonElement root, CancellationToken ct)
        {
            if (!TryGetRequiredString(root, "symbol", out string symbol, out string symbolErr))
                return Task.FromResult(BuildInvokerFailure(symbolErr));

            string? tf = TryGetOptionalString(root, "tf");
            string? range = TryGetOptionalString(root, "range");
            return Task.FromResult(_executionTools.OpenChartInternal(symbol, tf, range));
        }

        private async Task<McpToolResult> InvokeGetNewsHeadlinesAsync(
            JsonElement root,
            CancellationToken ct)
        {
            string symbol = TryGetOptionalString(root, "symbol") ?? "";

            int limit = 10;
            if (root.TryGetProperty("limit", out var limitProp))
            {
                if (!TryReadInt(limitProp, out limit))
                    return BuildInvokerFailure("Argument 'limit' must be an integer.");
            }

            string content = await _newsTools.GetNewsHeadlines(symbol, limit, ct);
            return InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content);
        }

        private async Task<McpToolResult> InvokeScrapeWebsiteTextAsync(
            JsonElement root,
            CancellationToken ct)
        {
            if (!TryGetRequiredString(root, "url", out string url, out string urlErr))
                return BuildInvokerFailure(urlErr);

            int timeoutSec = 12;
            if (root.TryGetProperty("timeoutSec", out var timeoutProp))
            {
                if (!TryReadInt(timeoutProp, out timeoutSec))
                    return BuildInvokerFailure("Argument 'timeoutSec' must be an integer.");
            }

            string content = await _newsTools.ScrapeWebsiteText(url, timeoutSec, ct);
            return InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content);
        }

        private static bool TryGetRequiredInt(
            JsonElement root,
            string property,
            out int value,
            out string error)
        {
            value = default;
            error = "";

            if (!root.TryGetProperty(property, out var prop))
            {
                error = $"Missing required argument '{property}'.";
                return false;
            }

            if (!TryReadInt(prop, out value))
            {
                error = $"Argument '{property}' must be an integer.";
                return false;
            }

            return true;
        }

        private static bool TryGetRequiredDecimal(
            JsonElement root,
            string property,
            out decimal value,
            out string error)
        {
            value = default;
            error = "";

            if (!root.TryGetProperty(property, out var prop))
            {
                error = $"Missing required argument '{property}'.";
                return false;
            }

            if (!TryReadDecimal(prop, out value))
            {
                error = $"Argument '{property}' must be a number.";
                return false;
            }

            return true;
        }

        private static bool TryGetRequiredString(
            JsonElement root,
            string property,
            out string value,
            out string error)
        {
            value = "";
            error = "";

            if (!root.TryGetProperty(property, out var prop))
            {
                error = $"Missing required argument '{property}'.";
                return false;
            }

            if (prop.ValueKind != JsonValueKind.String)
            {
                error = $"Argument '{property}' must be a string.";
                return false;
            }

            value = prop.GetString()?.Trim() ?? "";
            if (string.IsNullOrWhiteSpace(value))
            {
                error = $"Argument '{property}' cannot be empty.";
                return false;
            }

            return true;
        }

        private static string? TryGetOptionalString(JsonElement root, string property)
        {
            if (!root.TryGetProperty(property, out var prop))
                return null;

            if (prop.ValueKind == JsonValueKind.Null || prop.ValueKind == JsonValueKind.Undefined)
                return null;

            return prop.ValueKind == JsonValueKind.String ? prop.GetString() : null;
        }

        private static bool TryReadInt(JsonElement el, out int value)
        {
            value = default;

            return el.ValueKind switch
            {
                JsonValueKind.Number => el.TryGetInt32(out value),
                JsonValueKind.String => int.TryParse(
                    el.GetString(),
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out value),
                _ => false
            };
        }

        private static bool TryReadDecimal(JsonElement el, out decimal value)
        {
            value = default;

            return el.ValueKind switch
            {
                JsonValueKind.Number => el.TryGetDecimal(out value),
                JsonValueKind.String => decimal.TryParse(
                    el.GetString(),
                    NumberStyles.Number,
                    CultureInfo.InvariantCulture,
                    out value),
                _ => false
            };
        }

        private static bool InferSuccess(string content)
        {
            if (string.IsNullOrWhiteSpace(content))
                return false;

            if (content.StartsWith("ERROR", StringComparison.OrdinalIgnoreCase) ||
                content.StartsWith("SYSTEM ERROR", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            try
            {
                using var doc = JsonDocument.Parse(content);
                if (doc.RootElement.ValueKind == JsonValueKind.Object &&
                    doc.RootElement.TryGetProperty("ok", out var okProp) &&
                    okProp.ValueKind == JsonValueKind.False)
                {
                    return false;
                }
            }
            catch
            {
                // Non-JSON output is valid for execute_trade/open_chart.
            }

            return true;
        }

        private static McpToolResult BuildInvokerFailure(string message)
        {
            return McpToolResult.Failure(BuildErrorJson(message), message);
        }

        private static string BuildErrorJson(string message)
        {
            string escaped = message.Replace("\\", "\\\\").Replace("\"", "\\\"");
            return $"{{\"ok\":false,\"error\":\"{escaped}\"}}";
        }
    }
}
