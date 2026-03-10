namespace Aleph;

public sealed class Aether : IAether
{
    private const int MinDays = 1;
    private const int MaxDays = 365;
    private const int TimeoutMs = 30_000;

    private readonly IAxiom _axiom;
    private readonly ILogger<Aether> _logger;

    public Aether(IAxiom axiom, ILogger<Aether> logger)
    {
        _axiom = axiom;
        _logger = logger;
    }

    public async Task<AetherAnalysisResult> AnalyzeAsync(
        AetherAnalysisRequest request,
        CancellationToken ct = default)
    {
        if (request is null)
            throw new ArgumentNullException(nameof(request));

        if (!SymbolValidator.TryNormalize(request.Symbol, out var normalizedSymbol))
        {
            return new AetherAnalysisResult(false, string.Empty, "Invalid symbol format.", -1, false);
        }

        var days = Math.Clamp(request.Days, MinDays, MaxDays);
        var dataRoot = string.IsNullOrWhiteSpace(request.DataRoot)
            ? "data_lake/market/ohlcv"
            : request.DataRoot;

        var args = new List<string>
        {
            "--symbol", normalizedSymbol,
            "--days", days.ToString(),
            "--dataRoot", dataRoot
        };

        var routeResult = await _axiom.Python.RunJsonAsync("market", "parquet-read", args, TimeoutMs, ct);
        if (!routeResult.Success)
        {
            var error = routeResult.TimedOut
                ? "Parquet read timed out."
                : $"Parquet read failed (exit={routeResult.ExitCode}).";

            if (!string.IsNullOrWhiteSpace(routeResult.Stderr))
            {
                error = $"{error} {routeResult.Stderr}";
            }

            _logger.LogWarning("[Aether] Analyze failed for {Symbol}: {Error}", normalizedSymbol, error);
            return new AetherAnalysisResult(
                false,
                routeResult.StdoutJson,
                error,
                routeResult.ExitCode,
                routeResult.TimedOut);
        }

        if (string.IsNullOrWhiteSpace(routeResult.StdoutJson))
        {
            return new AetherAnalysisResult(
                false,
                string.Empty,
                "Parquet read returned empty stdout.",
                routeResult.ExitCode,
                routeResult.TimedOut);
        }

        return new AetherAnalysisResult(
            true,
            routeResult.StdoutJson,
            null,
            routeResult.ExitCode,
            routeResult.TimedOut);
    }
}
