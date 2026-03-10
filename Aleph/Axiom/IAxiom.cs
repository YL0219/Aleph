using System.Text.Json.Nodes;

namespace Aleph;

public interface IAxiom
{
    IAxiom.IPythonRouter Python { get; }
    IAxiom.IMcpGateway Mcp { get; }
    IAxiom.ITradeGateway Trades { get; }
    IAxiom.IChatGateway Chat { get; }
    IAxiom.IToolRunGateway ToolRuns { get; }
    IAxiom.ISkillGateway Skills { get; }

    public interface IPythonRouter
    {
        Task<PythonRouteResult> RunJsonAsync(
            string domain,
            string action,
            IReadOnlyList<string> arguments,
            int timeoutMs,
            CancellationToken ct = default);
    }

    public interface IMcpGateway
    {
        IReadOnlyList<JsonNode> GetOpenAiToolSchemas();

        bool IsStateChangingTool(string toolName);

        Task<McpInvokeResult> InvokeAsync(
            string toolName,
            string argumentsJson,
            CancellationToken ct = default);
    }

    public interface ITradeGateway
    {
        Task<TradeExecutionResult> ExecuteTradeAsync(
            ExecuteTradeRequest request,
            CancellationToken ct = default);
    }

    public interface IChatGateway
    {
        Task<IReadOnlyList<ChatMessageDto>> GetThreadMessagesAsync(
            string threadId,
            CancellationToken ct = default);

        Task AppendMessageAsync(
            ChatMessageWrite message,
            CancellationToken ct = default);
    }

    public interface IToolRunGateway
    {
        Task AppendBatchAsync(
            IReadOnlyList<ToolRunWrite> runs,
            CancellationToken ct = default);
    }

    public interface ISkillGateway
    {
        string GetAvailableSkills(bool includeDeprecated = false);

        string ReadPlaybook(string skillName);
    }
}

public sealed record PythonRouteResult(
    bool Success,
    string StdoutJson,
    string Stderr,
    int ExitCode,
    bool TimedOut);

public sealed record McpInvokeResult(
    string ToolContent,
    IReadOnlyList<object> UiActions,
    bool IsSuccess,
    string? Error);

public sealed record ChatMessageDto(
    string ThreadId,
    string Role,
    string Content,
    string? MetadataJson,
    DateTime CreatedAtUtc);

public sealed record ChatMessageWrite(
    string ThreadId,
    string Role,
    string Content,
    string? MetadataJson,
    DateTime? CreatedAtUtc = null);

public sealed record ToolRunWrite(
    string ThreadId,
    string ToolName,
    string ArgumentsJson,
    string ResultJson,
    long ExecutionTimeMs,
    bool IsSuccess,
    DateTime CreatedAtUtc);

public sealed record ExecuteTradeRequest
{
    public required string ClientRequestId { get; init; }
    public required string Symbol { get; init; }
    public required string Side { get; init; }
    public required decimal Quantity { get; init; }
    public required decimal ExecutedPrice { get; init; }
    public decimal? Fees { get; init; }
    public string? Currency { get; init; }
    public string? Notes { get; init; }
    public string? RawJson { get; init; }
}

public sealed record TradeExecutionResult(
    bool Success,
    bool IsDuplicate,
    bool IsConflict,
    TradeSnapshotDto? Trade,
    string? ErrorMessage);

public sealed record TradeSnapshotDto(
    int Id,
    string ClientRequestId,
    string Symbol,
    string Side,
    decimal Quantity,
    decimal ExecutedPrice,
    decimal? Fees,
    string? Currency,
    DateTime ExecutedAtUtc,
    string Status,
    string? Notes,
    string? RawJson,
    int? PositionId);
