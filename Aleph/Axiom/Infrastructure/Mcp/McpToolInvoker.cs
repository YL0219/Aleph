using System.Globalization;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;

namespace Aleph
{
    /// <summary>
    /// Single execution gateway for all MCP-backed AI tool calls.
    /// </summary>
    public sealed class McpToolInvoker
    {
        private sealed record ToolRoute(
            bool IsStateChanging,
            Func<JsonElement, CancellationToken, Task<McpToolResult>> Handler);

        private readonly IServiceProvider _serviceProvider;
        private readonly ILogger<McpToolInvoker> _logger;
        private readonly IReadOnlyDictionary<string, ToolRoute> _routes;
        private readonly IReadOnlySet<string> _stateChangingTools;

        public McpToolInvoker(
            IServiceProvider serviceProvider,
            ILogger<McpToolInvoker> logger)
        {
            _serviceProvider = serviceProvider;
            _logger = logger;

            var routes = new Dictionary<string, ToolRoute>(StringComparer.OrdinalIgnoreCase)
            {
                ["query_local_market_data"] = new(
                    IsStateChanging: false,
                    Handler: InvokeQueryLocalMarketDataAsync),
                ["perceive_market_data"] = new(
                    IsStateChanging: false,
                    Handler: InvokePerceiveMarketDataAsync),
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
                    Handler: InvokeScrapeWebsiteTextAsync),
                ["get_available_skills"] = new(
                    IsStateChanging: false,
                    Handler: InvokeGetAvailableSkillsAsync),
                ["read_skill_playbook"] = new(
                    IsStateChanging: false,
                    Handler: InvokeReadSkillPlaybookAsync),
                ["aether_get_status"] = new(
                    IsStateChanging: false,
                    Handler: InvokeAetherGetStatusAsync),
                ["aether_math_run"] = new(
                    IsStateChanging: false,
                    Handler: InvokeAetherMathRunAsync),
                ["aether_ml_predict"] = new(
                    IsStateChanging: false,
                    Handler: InvokeAetherMlPredictAsync),
                ["aether_ml_train"] = new(
                    IsStateChanging: true,
                    Handler: InvokeAetherMlTrainAsync),
                ["aether_sim_run"] = new(
                    IsStateChanging: false,
                    Handler: InvokeAetherSimRunAsync),
                ["aether_macro_check"] = new(
                    IsStateChanging: false,
                    Handler: InvokeAetherMacroCheckAsync),
                ["aether_release_adrenaline"] = new(
                    IsStateChanging: true,
                    Handler: InvokeAetherReleaseAdrenalineAsync)
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

            var marketTools = ResolveTool<McpMarketTools>();
            string content = await marketTools.QueryLocalMarketData(symbol, days);
            return InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content);
        }

        private async Task<McpToolResult> InvokePerceiveMarketDataAsync(
            JsonElement root,
            CancellationToken ct)
        {
            if (!TryGetRequiredString(root, "symbol", out string symbol, out string err))
                return BuildInvokerFailure(err);

            int days = 90;
            if (root.TryGetProperty("days", out var daysProp))
            {
                if (!TryReadInt(daysProp, out days))
                    return BuildInvokerFailure("Argument 'days' must be an integer.");
            }

            string interval = TryGetOptionalString(root, "interval") ?? "1d";

            var marketTools = ResolveTool<McpMarketTools>();
            string content = await marketTools.PerceiveMarketData(symbol, days, interval);
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

            var executionTools = ResolveTool<McpExecutionTools>();
            return executionTools.ExecuteTradeInternalAsync(action, symbol, shares, price, ct);
        }

        private Task<McpToolResult> InvokeOpenChartAsync(JsonElement root, CancellationToken ct)
        {
            if (!TryGetRequiredString(root, "symbol", out string symbol, out string symbolErr))
                return Task.FromResult(BuildInvokerFailure(symbolErr));

            string? tf = TryGetOptionalString(root, "tf");
            string? range = TryGetOptionalString(root, "range");
            var executionTools = ResolveTool<McpExecutionTools>();
            return Task.FromResult(executionTools.OpenChartInternal(symbol, tf, range));
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

            var newsTools = ResolveTool<McpNewsTools>();
            string content = await newsTools.GetNewsHeadlines(symbol, limit, ct);
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

            var newsTools = ResolveTool<McpNewsTools>();
            string content = await newsTools.ScrapeWebsiteText(url, timeoutSec, ct);
            return InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content);
        }

        // ─── Skill Tools ──────────────────────────────────────────────────

        private Task<McpToolResult> InvokeGetAvailableSkillsAsync(
            JsonElement root,
            CancellationToken ct)
        {
            bool includeDeprecated = false;
            if (root.TryGetProperty("include_deprecated", out var prop) &&
                prop.ValueKind == JsonValueKind.True)
            {
                includeDeprecated = true;
            }

            var skillTools = ResolveTool<McpSkillTools>();
            string content = skillTools.GetAvailableSkills(includeDeprecated);
            return Task.FromResult(McpToolResult.Success(content));
        }

        private Task<McpToolResult> InvokeReadSkillPlaybookAsync(
            JsonElement root,
            CancellationToken ct)
        {
            if (!TryGetRequiredString(root, "skill_name", out string skillName, out string err))
                return Task.FromResult(BuildInvokerFailure(err));

            var skillTools = ResolveTool<McpSkillTools>();
            string content = skillTools.ReadSkillPlaybook(skillName);
            return Task.FromResult(InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content));
        }

        private async Task<McpToolResult> InvokeAetherGetStatusAsync(
            JsonElement root,
            CancellationToken ct)
        {
            string symbol = TryGetOptionalString(root, "symbol") ?? "";
            var aetherTools = ResolveTool<McpAetherTools>();
            string content = await aetherTools.AetherGetStatus(symbol, ct);
            return InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content);
        }

        private async Task<McpToolResult> InvokeAetherMathRunAsync(
            JsonElement root,
            CancellationToken ct)
        {
            if (!TryGetRequiredString(root, "symbol", out string symbol, out string err))
                return BuildInvokerFailure(err);

            int days = 30;
            if (root.TryGetProperty("days", out var daysProp))
            {
                if (!TryReadInt(daysProp, out days))
                    return BuildInvokerFailure("Argument 'days' must be an integer.");
            }

            var aetherTools = ResolveTool<McpAetherTools>();
            string content = await aetherTools.AetherMathRun(symbol, days, ct);
            return InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content);
        }

        private async Task<McpToolResult> InvokeAetherMlPredictAsync(
            JsonElement root,
            CancellationToken ct)
        {
            if (!TryGetRequiredString(root, "symbol", out string symbol, out string err))
                return BuildInvokerFailure(err);

            int horizonDays = 5;
            if (root.TryGetProperty("horizon_days", out var hdProp))
            {
                if (!TryReadInt(hdProp, out horizonDays))
                    return BuildInvokerFailure("Argument 'horizon_days' must be an integer.");
            }

            var aetherTools = ResolveTool<McpAetherTools>();
            string content = await aetherTools.AetherMlPredict(symbol, horizonDays, ct);
            return InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content);
        }

        private async Task<McpToolResult> InvokeAetherMlTrainAsync(
            JsonElement root,
            CancellationToken ct)
        {
            if (!TryGetRequiredString(root, "symbol", out string symbol, out string err))
                return BuildInvokerFailure(err);

            int epochs = 1;
            if (root.TryGetProperty("epochs", out var epochsProp))
            {
                if (!TryReadInt(epochsProp, out epochs))
                    return BuildInvokerFailure("Argument 'epochs' must be an integer.");
            }

            var aetherTools = ResolveTool<McpAetherTools>();
            string content = await aetherTools.AetherMlTrain(symbol, epochs, ct);
            return InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content);
        }

        private async Task<McpToolResult> InvokeAetherSimRunAsync(
            JsonElement root,
            CancellationToken ct)
        {
            if (!TryGetRequiredString(root, "symbol", out string symbol, out string err))
                return BuildInvokerFailure(err);

            int days = 180;
            if (root.TryGetProperty("days", out var daysProp))
            {
                if (!TryReadInt(daysProp, out days))
                    return BuildInvokerFailure("Argument 'days' must be an integer.");
            }

            string strategy = TryGetOptionalString(root, "strategy") ?? "baseline";

            var aetherTools = ResolveTool<McpAetherTools>();
            string content = await aetherTools.AetherSimRun(symbol, days, strategy, ct);
            return InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content);
        }

        private async Task<McpToolResult> InvokeAetherMacroCheckAsync(
            JsonElement root,
            CancellationToken ct)
        {
            string region = TryGetOptionalString(root, "region") ?? "global";
            var aetherTools = ResolveTool<McpAetherTools>();
            string content = await aetherTools.AetherMacroCheck(region, ct);
            return InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content);
        }

        private async Task<McpToolResult> InvokeAetherReleaseAdrenalineAsync(
            JsonElement root,
            CancellationToken ct)
        {
            if (!TryGetRequiredString(root, "source", out string source, out string sourceErr))
                return BuildInvokerFailure(sourceErr);

            if (!TryGetRequiredInt(root, "severity", out int severity, out string sevErr))
                return BuildInvokerFailure(sevErr);

            string message = TryGetOptionalString(root, "message") ?? "";
            string tags = TryGetOptionalString(root, "tags") ?? "";

            int ttlSeconds = 0;
            if (root.TryGetProperty("ttl_seconds", out var ttlProp))
            {
                if (!TryReadInt(ttlProp, out ttlSeconds))
                    return BuildInvokerFailure("Argument 'ttl_seconds' must be an integer.");
            }

            var aetherTools = ResolveTool<McpAetherTools>();
            string content = await aetherTools.AetherReleaseAdrenaline(
                source, severity, message, tags, ttlSeconds, ct);
            return InferSuccess(content)
                ? McpToolResult.Success(content)
                : McpToolResult.Failure(content);
        }

        // ─── Argument Parsing Helpers ─────────────────────────────────────

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

        private TTool ResolveTool<TTool>() where TTool : notnull
        {
            return _serviceProvider.GetRequiredService<TTool>();
        }
    }
}

