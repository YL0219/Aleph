using System.ComponentModel;
using ModelContextProtocol.Server;

namespace Aleph
{
    [McpServerToolType]
    public sealed class McpAetherTools
    {
        private readonly IAether _aether;
        private readonly ILogger<McpAetherTools> _logger;

        public McpAetherTools(
            IAether aether,
            ILogger<McpAetherTools> logger)
        {
            _aether = aether;
            _logger = logger;
        }

        [McpServerTool(Name = "aether_get_status", ReadOnly = true)]
        [Description("Get Aether ML engine status as JSON. Optional symbol narrows the status scope.")]
        public async Task<string> AetherGetStatus(
            [Description("Optional market symbol, e.g. SI=F or AAPL.")]
            string symbol = "",
            CancellationToken ct = default)
        {
            var result = await _aether.Ml.GetStatusAsync(new MlStatusRequest(symbol), ct);
            return ToToolJson("aether_get_status", result);
        }

        [McpServerTool(Name = "aether_math_run", ReadOnly = true)]
        [Description("Run Aether quantitative indicator evaluation for a symbol.")]
        public async Task<string> AetherMathRun(
            [Description("Market symbol (e.g. SI=F, GC=F, AAPL).")]
            string symbol,
            [Description("Historical window in days. Default is 30.")]
            int days = 30,
            CancellationToken ct = default)
        {
            var request = new MathIndicatorsRequest(symbol, days);
            var result = await _aether.Math.EvaluateIndicatorsAsync(request, ct);
            return ToToolJson("aether_math_run", result);
        }

        [McpServerTool(Name = "aether_ml_predict", ReadOnly = true)]
        [Description("Run Aether ML prediction placeholder for a symbol.")]
        public async Task<string> AetherMlPredict(
            [Description("Market symbol (e.g. SI=F, GC=F, AAPL).")]
            string symbol,
            [Description("Prediction horizon in days. Default is 5.")]
            int horizon_days = 5,
            CancellationToken ct = default)
        {
            var request = new MlPredictRequest(symbol, horizon_days);
            var result = await _aether.Ml.PredictAsync(request, ct);
            return ToToolJson("aether_ml_predict", result);
        }

        [McpServerTool(Name = "aether_ml_train", ReadOnly = false)]
        [Description("Trigger Aether ML training placeholder for a symbol.")]
        public async Task<string> AetherMlTrain(
            [Description("Market symbol (e.g. SI=F, GC=F, AAPL).")]
            string symbol,
            [Description("Epoch count. Default is 1.")]
            int epochs = 1,
            CancellationToken ct = default)
        {
            var request = new MlTrainRequest(symbol, epochs);
            var result = await _aether.Ml.TrainAsync(request, ct);
            return ToToolJson("aether_ml_train", result);
        }

        [McpServerTool(Name = "aether_sim_run", ReadOnly = true)]
        [Description("Run Aether simulation/backtest placeholder for a symbol.")]
        public async Task<string> AetherSimRun(
            [Description("Market symbol (e.g. SI=F, GC=F, AAPL).")]
            string symbol,
            [Description("Backtest window in days. Default is 180.")]
            int days = 180,
            [Description("Simulation strategy name. Default is 'baseline'.")]
            string strategy = "baseline",
            CancellationToken ct = default)
        {
            var request = new SimBacktestRequest(symbol, days, strategy);
            var result = await _aether.Sim.RunBacktestAsync(request, ct);
            return ToToolJson("aether_sim_run", result);
        }

        [McpServerTool(Name = "aether_macro_check", ReadOnly = true)]
        [Description("Run Aether macro regime placeholder check.")]
        public async Task<string> AetherMacroCheck(
            [Description("Macro region key (e.g. global, us, eu).")]
            string region = "global",
            CancellationToken ct = default)
        {
            var result = await _aether.Macro.CheckRegimeAsync(new MacroRegimeRequest(region), ct);
            return ToToolJson("aether_macro_check", result);
        }

        [McpServerTool(Name = "aether_release_adrenaline", ReadOnly = false)]
        [Description("Inject conscious stress/adrenaline into Aether's autonomic system. " +
            "Use this when the AI detects danger, market panic, or needs to raise system alertness. " +
            "Severity: 0=Normal, 1=Elevated, 2=Warning, 3=Critical.")]
        public async Task<string> AetherReleaseAdrenaline(
            [Description("Identifying source of the stress signal, e.g. 'arbiter', 'user', 'news_alert'.")]
            string source,
            [Description("Severity level: 0=Normal, 1=Elevated, 2=Warning, 3=Critical.")]
            int severity,
            [Description("Human-readable description of the stress trigger.")]
            string message = "",
            [Description("Optional comma-separated tags, e.g. 'market_crash,spy_drop'.")]
            string tags = "",
            [Description("Optional TTL in seconds. Natural decay is primary mechanism; TTL is advisory.")]
            int ttl_seconds = 0,
            CancellationToken ct = default)
        {
            // Validate severity range
            var parsedSeverity = (PulseSeverity)Math.Clamp(severity, 0, 3);

            if (string.IsNullOrWhiteSpace(source))
            {
                _logger.LogWarning("[MCP/Aether] aether_release_adrenaline: empty source rejected.");
                return BuildErrorJson("Source must be provided.");
            }

            var parsedTags = string.IsNullOrWhiteSpace(tags)
                ? null
                : (IReadOnlyList<string>)tags.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .ToList();

            var request = new AdrenalineRequest
            {
                Source = source.Trim(),
                Severity = parsedSeverity,
                Message = string.IsNullOrWhiteSpace(message) ? null : message.Trim(),
                Tags = parsedTags,
                TtlSeconds = ttl_seconds > 0 ? ttl_seconds : null
            };

            var result = await _aether.Regulation.ReleaseAdrenalineAsync(request, ct);

            return System.Text.Json.JsonSerializer.Serialize(new
            {
                ok = result.Accepted,
                source = result.Source,
                severity = result.Severity.ToString(),
                timestamp_utc = result.TimestampUtc.ToString("o"),
                rejection_reason = result.RejectionReason
            });
        }

        private string ToToolJson(string toolName, AetherJsonResult result)
        {
            if (!string.IsNullOrWhiteSpace(result.PayloadJson))
            {
                return result.PayloadJson;
            }

            _logger.LogWarning("[MCP/Aether] Tool {ToolName} returned empty payload. Error={Error}", toolName, result.Error);
            return BuildErrorJson(result.Error ?? "Aether returned empty payload.");
        }

        private static string BuildErrorJson(string message)
        {
            var escaped = message.Replace("\\", "\\\\").Replace("\"", "\\\"");
            return $"{{\"ok\":false,\"error\":\"{escaped}\"}}";
        }
    }
}

