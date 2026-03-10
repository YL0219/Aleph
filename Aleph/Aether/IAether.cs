namespace Aleph;

public interface IAether
{
    Task<AetherAnalysisResult> AnalyzeAsync(AetherAnalysisRequest request, CancellationToken ct = default);
}

public sealed record AetherAnalysisRequest(
    string Symbol,
    int Days = 30,
    string DataRoot = "data_lake/market/ohlcv");

public sealed record AetherAnalysisResult(
    bool Success,
    string PayloadJson,
    string? Error,
    int ExitCode,
    bool TimedOut);
