namespace Aleph;

public interface IArbiter
{
    Task<ArbiterHandleResult> HandleAsync(ChatRequest request, CancellationToken ct = default);
}

public sealed record ArbiterHandleResult(
    string Response,
    IReadOnlyList<object> UiActions,
    bool TerminatedByCircuitBreaker,
    int Iterations);
