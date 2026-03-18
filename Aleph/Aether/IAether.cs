namespace Aleph;

public interface IAether
{
    IAether.IMathGateway Math { get; }
    IAether.IMlGateway Ml { get; }
    IAether.ISimGateway Sim { get; }
    IAether.IMacroGateway Macro { get; }
    IAether.IRegulationGateway Regulation { get; }

    public interface IMathGateway
    {
        Task<AetherJsonResult> EvaluateIndicatorsAsync(MathIndicatorsRequest request, CancellationToken ct = default);
    }

    public interface IMlGateway
    {
        Task<AetherJsonResult> PredictAsync(MlPredictRequest request, CancellationToken ct = default);
        Task<AetherJsonResult> TrainAsync(MlTrainRequest request, CancellationToken ct = default);
        Task<AetherJsonResult> GetStatusAsync(MlStatusRequest request, CancellationToken ct = default);

        /// <summary>
        /// Cortex real-time prediction from a metabolic payload.
        /// Lightweight path — always available, even during cold start.
        /// </summary>
        Task<AetherJsonResult> CortexPredictAsync(MlCortexPredictRequest request, CancellationToken ct = default);

        /// <summary>
        /// Cortex incremental training update. Heavier path — should only be
        /// called during Calm/DeepWork windows via Homeostasis gating.
        /// </summary>
        Task<AetherJsonResult> CortexTrainAsync(MlCortexTrainRequest request, CancellationToken ct = default);

        /// <summary>
        /// Cortex status — model state, sample counts, version info.
        /// </summary>
        Task<AetherJsonResult> CortexStatusAsync(MlCortexStatusRequest request, CancellationToken ct = default);

        /// <summary>
        /// Cortex resolve — scan pending memory, resolve mature predictions
        /// against parquet truth, produce labeled training data.
        /// Should only be called during sleep/calm windows.
        /// </summary>
        Task<AetherJsonResult> CortexResolveAsync(MlCortexResolveRequest request, CancellationToken ct = default);
    }

    public interface ISimGateway
    {
        Task<AetherJsonResult> RunBacktestAsync(SimBacktestRequest request, CancellationToken ct = default);
    }

    public interface IMacroGateway
    {
        Task<AetherJsonResult> CheckRegimeAsync(MacroRegimeRequest request, CancellationToken ct = default);
    }

    public interface IRegulationGateway
    {
        /// <summary>
        /// Conscious adrenaline release — Arbiter or MCP tools call this.
        /// Builds PulseEnvelope.ExternalStress and injects through IStressInjector.
        /// </summary>
        Task<AdrenalineReleaseResult> ReleaseAdrenalineAsync(AdrenalineRequest request, CancellationToken ct = default);
    }
}

public sealed record AetherJsonResult(
    bool Success,
    string PayloadJson,
    string? Error,
    int ExitCode,
    bool TimedOut);

public sealed record MathIndicatorsRequest(
    string Symbol,
    int Days = 30,
    string Timeframe = "1d");

public sealed record MlPredictRequest(
    string Symbol,
    int HorizonDays = 5);

public sealed record MlTrainRequest(
    string Symbol,
    int Epochs = 1);

public sealed record MlStatusRequest(
    string? Symbol = null);

public sealed record SimBacktestRequest(
    string Symbol,
    int Days = 180,
    string Strategy = "baseline");

public sealed record MacroRegimeRequest(
    string Region = "global");

// ─── Regulation DTOs ─────────────────────────────────────────────

public sealed record AdrenalineRequest
{
    public required string Source { get; init; }
    public required PulseSeverity Severity { get; init; }
    public string? Message { get; init; }
    public IReadOnlyList<string>? Tags { get; init; }
    public IReadOnlyDictionary<string, double>? Metrics { get; init; }
    public int? TtlSeconds { get; init; }
}

public sealed record AdrenalineReleaseResult
{
    public required bool Accepted { get; init; }
    public required string Source { get; init; }
    public required PulseSeverity Severity { get; init; }
    public required DateTimeOffset TimestampUtc { get; init; }
    public string? RejectionReason { get; init; }
};

// ─── ML Cortex DTOs ─────────────────────────────────────────────

/// <summary>
/// Request for Cortex real-time prediction from a MetabolicEvent payload.
/// The payload JSON is the serialized metabolic feature snapshot.
/// </summary>
public sealed record MlCortexPredictRequest
{
    public required string Symbol { get; init; }
    public required string Interval { get; init; }
    public required string ActiveHorizon { get; init; }
    public required string AsOfUtc { get; init; }
    public required string MetabolicPayloadJson { get; init; }
}

/// <summary>
/// Request for Cortex incremental training update.
/// </summary>
public sealed record MlCortexTrainRequest
{
    public required string Symbol { get; init; }
    public string ActiveHorizon { get; init; } = "1d";
    public int MaxSamples { get; init; } = 100;
}

/// <summary>
/// Request for Cortex status information.
/// </summary>
public sealed record MlCortexStatusRequest
{
    public string? Symbol { get; init; }
    public string ActiveHorizon { get; init; } = "1d";
}

/// <summary>
/// Request for Cortex resolve — resolve pending predictions against parquet truth.
/// </summary>
public sealed record MlCortexResolveRequest
{
    public required string Symbol { get; init; }
    public string ActiveHorizon { get; init; } = "1d";
    public string Interval { get; init; } = "1h";
}
