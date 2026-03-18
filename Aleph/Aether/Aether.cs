using System.Text.RegularExpressions;

namespace Aleph;

public sealed class Aether : IAether
{
    private const int DefaultTimeoutMs = 30_000;

    private static readonly Regex SymbolPattern =
        new("^[A-Z0-9][A-Z0-9.-]{0,14}$", RegexOptions.Compiled);

    private readonly IAxiom _axiom;
    private readonly IStressInjector _stressInjector;
    private readonly ILogger<Aether> _logger;

    public IAether.IMathGateway Math { get; }
    public IAether.IMlGateway Ml { get; }
    public IAether.ISimGateway Sim { get; }
    public IAether.IMacroGateway Macro { get; }
    public IAether.IRegulationGateway Regulation { get; }

    public Aether(IAxiom axiom, IStressInjector stressInjector, ILogger<Aether> logger)
    {
        _axiom = axiom;
        _stressInjector = stressInjector;
        _logger = logger;

        Math = new MathGateway(this);
        Ml = new MlGateway(this);
        Sim = new SimGateway(this);
        Macro = new MacroGateway(this);
        Regulation = new RegulationGateway(this);
    }

    private async Task<AetherJsonResult> RunAsync(
        string domain,
        string action,
        IReadOnlyList<string> arguments,
        int timeoutMs,
        CancellationToken ct)
    {
        var routeArgs = new List<string> { action };
        routeArgs.AddRange(arguments);

        var routeResult = await _axiom.Python.RunJsonAsync("aether", domain, routeArgs, timeoutMs, ct);

        if (!routeResult.Success)
        {
            var error = routeResult.TimedOut
                ? $"Aether route timed out: {domain}/{action}."
                : $"Aether route failed: {domain}/{action} (exit={routeResult.ExitCode}).";

            if (!string.IsNullOrWhiteSpace(routeResult.Stderr))
            {
                error = $"{error} {routeResult.Stderr}";
            }

            _logger.LogWarning("[Aether] Route failure {Domain}/{Action}: {Error}", domain, action, error);
            return new AetherJsonResult(
                false,
                routeResult.StdoutJson,
                error,
                routeResult.ExitCode,
                routeResult.TimedOut);
        }

        if (string.IsNullOrWhiteSpace(routeResult.StdoutJson))
        {
            return new AetherJsonResult(
                false,
                string.Empty,
                "Aether route returned empty stdout.",
                routeResult.ExitCode,
                routeResult.TimedOut);
        }

        return new AetherJsonResult(
            true,
            routeResult.StdoutJson,
            null,
            routeResult.ExitCode,
            routeResult.TimedOut);
    }

    private static string NormalizeSymbol(string symbol)
    {
        var normalized = (symbol ?? string.Empty).Trim().ToUpperInvariant();
        return SymbolPattern.IsMatch(normalized) ? normalized : string.Empty;
    }

    private static string NormalizeRegion(string region)
    {
        var normalized = (region ?? string.Empty).Trim().ToLowerInvariant();
        return string.IsNullOrWhiteSpace(normalized) ? "global" : normalized;
    }

    private sealed class MathGateway : IAether.IMathGateway
    {
        private readonly Aether _root;

        public MathGateway(Aether root)
        {
            _root = root;
        }

        public Task<AetherJsonResult> EvaluateIndicatorsAsync(MathIndicatorsRequest request, CancellationToken ct = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));

            var symbol = NormalizeSymbol(request.Symbol);
            if (string.IsNullOrWhiteSpace(symbol))
                return Task.FromResult(new AetherJsonResult(false, string.Empty, "Invalid symbol format.", -1, false));

            var args = new List<string>
            {
                "--symbol", symbol,
                "--days", System.Math.Clamp(request.Days, 1, 3650).ToString(),
                "--timeframe", string.IsNullOrWhiteSpace(request.Timeframe) ? "1d" : request.Timeframe.Trim().ToLowerInvariant()
            };

            return _root.RunAsync("math", "indicators", args, DefaultTimeoutMs, ct);
        }
    }

    private sealed class MlGateway : IAether.IMlGateway
    {
        private readonly Aether _root;

        public MlGateway(Aether root)
        {
            _root = root;
        }

        public Task<AetherJsonResult> PredictAsync(MlPredictRequest request, CancellationToken ct = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));

            var symbol = NormalizeSymbol(request.Symbol);
            if (string.IsNullOrWhiteSpace(symbol))
                return Task.FromResult(new AetherJsonResult(false, string.Empty, "Invalid symbol format.", -1, false));

            var args = new List<string>
            {
                "--symbol", symbol,
                "--horizonDays", System.Math.Clamp(request.HorizonDays, 1, 365).ToString()
            };

            return _root.RunAsync("ml", "predict", args, DefaultTimeoutMs, ct);
        }

        public Task<AetherJsonResult> TrainAsync(MlTrainRequest request, CancellationToken ct = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));

            var symbol = NormalizeSymbol(request.Symbol);
            if (string.IsNullOrWhiteSpace(symbol))
                return Task.FromResult(new AetherJsonResult(false, string.Empty, "Invalid symbol format.", -1, false));

            var args = new List<string>
            {
                "--symbol", symbol,
                "--epochs", System.Math.Clamp(request.Epochs, 1, 1000).ToString()
            };

            return _root.RunAsync("ml", "train", args, 120_000, ct);
        }

        public Task<AetherJsonResult> GetStatusAsync(MlStatusRequest request, CancellationToken ct = default)
        {
            request ??= new MlStatusRequest();

            var args = new List<string>();
            var symbol = NormalizeSymbol(request.Symbol ?? string.Empty);
            if (!string.IsNullOrWhiteSpace(symbol))
            {
                args.Add("--symbol");
                args.Add(symbol);
            }

            return _root.RunAsync("ml", "status", args, DefaultTimeoutMs, ct);
        }

        public Task<AetherJsonResult> CortexPredictAsync(MlCortexPredictRequest request, CancellationToken ct = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));

            var symbol = NormalizeSymbol(request.Symbol);
            if (string.IsNullOrWhiteSpace(symbol))
                return Task.FromResult(new AetherJsonResult(false, string.Empty, "Invalid symbol format.", -1, false));

            var args = new List<string>
            {
                "--symbol", symbol,
                "--interval", string.IsNullOrWhiteSpace(request.Interval) ? "1d" : request.Interval.Trim().ToLowerInvariant(),
                "--horizon", string.IsNullOrWhiteSpace(request.ActiveHorizon) ? "1d" : request.ActiveHorizon.Trim().ToLowerInvariant(),
                "--asof", request.AsOfUtc ?? DateTimeOffset.UtcNow.ToString("o"),
                "--payload", request.MetabolicPayloadJson ?? "{}"
            };

            return _root.RunAsync("ml", "cortex_predict", args, DefaultTimeoutMs, ct);
        }

        public Task<AetherJsonResult> CortexTrainAsync(MlCortexTrainRequest request, CancellationToken ct = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));

            var symbol = NormalizeSymbol(request.Symbol);
            if (string.IsNullOrWhiteSpace(symbol))
                return Task.FromResult(new AetherJsonResult(false, string.Empty, "Invalid symbol format.", -1, false));

            var args = new List<string>
            {
                "--symbol", symbol,
                "--horizon", string.IsNullOrWhiteSpace(request.ActiveHorizon) ? "1d" : request.ActiveHorizon.Trim().ToLowerInvariant(),
                "--max-samples", System.Math.Clamp(request.MaxSamples, 1, 10000).ToString()
            };

            return _root.RunAsync("ml", "cortex_train", args, 120_000, ct);
        }

        public Task<AetherJsonResult> CortexStatusAsync(MlCortexStatusRequest request, CancellationToken ct = default)
        {
            request ??= new MlCortexStatusRequest();

            var args = new List<string>
            {
                "--horizon", string.IsNullOrWhiteSpace(request.ActiveHorizon) ? "1d" : request.ActiveHorizon.Trim().ToLowerInvariant()
            };

            var symbol = NormalizeSymbol(request.Symbol ?? string.Empty);
            if (!string.IsNullOrWhiteSpace(symbol))
            {
                args.Add("--symbol");
                args.Add(symbol);
            }

            return _root.RunAsync("ml", "cortex_status", args, DefaultTimeoutMs, ct);
        }

        public Task<AetherJsonResult> CortexResolveAsync(MlCortexResolveRequest request, CancellationToken ct = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));

            var symbol = NormalizeSymbol(request.Symbol);
            if (string.IsNullOrWhiteSpace(symbol))
                return Task.FromResult(new AetherJsonResult(false, string.Empty, "Invalid symbol format.", -1, false));

            var args = new List<string>
            {
                "--symbol", symbol,
                "--horizon", string.IsNullOrWhiteSpace(request.ActiveHorizon) ? "1d" : request.ActiveHorizon.Trim().ToLowerInvariant(),
                "--interval", string.IsNullOrWhiteSpace(request.Interval) ? "1h" : request.Interval.Trim().ToLowerInvariant()
            };

            return _root.RunAsync("ml", "cortex_resolve", args, 120_000, ct);
        }
    }

    private sealed class SimGateway : IAether.ISimGateway
    {
        private readonly Aether _root;

        public SimGateway(Aether root)
        {
            _root = root;
        }

        public Task<AetherJsonResult> RunBacktestAsync(SimBacktestRequest request, CancellationToken ct = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));

            var symbol = NormalizeSymbol(request.Symbol);
            if (string.IsNullOrWhiteSpace(symbol))
                return Task.FromResult(new AetherJsonResult(false, string.Empty, "Invalid symbol format.", -1, false));

            var strategy = string.IsNullOrWhiteSpace(request.Strategy)
                ? "baseline"
                : request.Strategy.Trim().ToLowerInvariant();

            var args = new List<string>
            {
                "--symbol", symbol,
                "--days", System.Math.Clamp(request.Days, 1, 3650).ToString(),
                "--strategy", strategy
            };

            return _root.RunAsync("sim", "backtest", args, 90_000, ct);
        }
    }

    private sealed class MacroGateway : IAether.IMacroGateway
    {
        private readonly Aether _root;

        public MacroGateway(Aether root)
        {
            _root = root;
        }

        public Task<AetherJsonResult> CheckRegimeAsync(MacroRegimeRequest request, CancellationToken ct = default)
        {
            request ??= new MacroRegimeRequest();

            var args = new List<string>
            {
                "--region", NormalizeRegion(request.Region)
            };

            return _root.RunAsync("macro", "regime", args, DefaultTimeoutMs, ct);
        }
    }

    private sealed class RegulationGateway : IAether.IRegulationGateway
    {
        private readonly Aether _root;

        public RegulationGateway(Aether root)
        {
            _root = root;
        }

        public Task<AdrenalineReleaseResult> ReleaseAdrenalineAsync(AdrenalineRequest request, CancellationToken ct = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));

            if (string.IsNullOrWhiteSpace(request.Source))
            {
                return Task.FromResult(new AdrenalineReleaseResult
                {
                    Accepted = false,
                    Source = "unknown",
                    Severity = request.Severity,
                    TimestampUtc = DateTimeOffset.UtcNow,
                    RejectionReason = "Source must be provided."
                });
            }

            // Build the PulseEnvelope via the ExternalStress factory
            var mergedTags = new List<string> { "conscious", $"src:{request.Source}" };
            if (request.Tags is not null)
                mergedTags.AddRange(request.Tags);

            var mergedMetrics = new Dictionary<string, double>();
            if (request.Metrics is not null)
            {
                foreach (var kv in request.Metrics)
                    mergedMetrics[kv.Key] = kv.Value;
            }
            if (request.TtlSeconds.HasValue)
                mergedMetrics["ttl_seconds"] = request.TtlSeconds.Value;

            var envelope = PulseEnvelope.ExternalStress(
                source: $"aether_regulation/{request.Source}",
                severity: request.Severity,
                message: request.Message,
                metrics: mergedMetrics.Count > 0 ? mergedMetrics : null,
                tags: mergedTags);

            var receipt = _root._stressInjector.InjectStress(envelope);

            _root._logger.LogInformation(
                "[Aether] Conscious adrenaline release. Source={Source}, Severity={Severity}, Accepted={Accepted}",
                request.Source, request.Severity, receipt.Accepted);

            return Task.FromResult(new AdrenalineReleaseResult
            {
                Accepted = receipt.Accepted,
                Source = request.Source,
                Severity = request.Severity,
                TimestampUtc = receipt.TimestampUtc,
                RejectionReason = receipt.RejectionReason
            });
        }
    }
}
