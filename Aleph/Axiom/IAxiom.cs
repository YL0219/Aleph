using System.Text.Json.Nodes;
using System.Text.Json;

namespace Aleph;

public interface IAxiom
{
    IAxiom.IPythonRouter Python { get; }
    IAxiom.IMarketGateway Market { get; }
    IAxiom.IMarketIngestionGateway MarketIngestion { get; }
    IAxiom.IMcpGateway Mcp { get; }
    IAxiom.ITradeGateway Trades { get; }
    IAxiom.IChatGateway Chat { get; }
    IAxiom.IToolRunGateway ToolRuns { get; }
    IAxiom.ISkillGateway Skills { get; }
    IAxiom.IPerceptionGateway Perception { get; }

    public interface IPythonRouter
    {
        Task<PythonRouteResult> RunJsonAsync(
            string domain,
            string action,
            IReadOnlyList<string> arguments,
            int timeoutMs,
            CancellationToken ct = default);
    }

    public interface IMarketGateway
    {
        Task<MarketQuoteFetchResult> GetQuoteAsync(
            string normalizedSymbol,
            CancellationToken ct = default);

        Task<MarketCandlesFetchResult> GetCandlesAsync(
            MarketCandlesQuery query,
            CancellationToken ct = default);

        /// <summary>
        /// On-demand perception: get-or-fetch a live quote with local-first + live overlay.
        /// Never adds the symbol to the background watchlist.
        /// </summary>
        Task<MarketPerceptionResult> PerceiveQuoteAsync(
            string symbol,
            CancellationToken ct = default);

        /// <summary>
        /// On-demand perception: get-or-fetch candle history with local-first + live bootstrap.
        /// If local history is missing, performs a live fetch and persists to data lake.
        /// Never adds the symbol to the background watchlist.
        /// </summary>
        Task<MarketPerceptionResult> PerceiveCandlesAsync(
            MarketPerceptionRequest request,
            CancellationToken ct = default);
    }

    public interface IMarketIngestionGateway
    {
        bool IsPythonAvailable { get; }

        Task<IReadOnlyList<string>> GetActiveSymbolsAsync(
            CancellationToken ct = default);

        Task<MarketIngestionRunResult> RunIngestionBatchAsync(
            IReadOnlyList<string> symbols,
            string interval,
            int lookbackDays,
            string outRoot,
            CancellationToken ct = default);

        Task ApplyIngestionBatchAsync(
            IReadOnlyList<string> batchSymbols,
            string interval,
            MarketIngestionRunResult runResult,
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

    public interface IPerceptionGateway
    {
        /// <summary>
        /// Run perception ingest — fetches macro proxies, economic calendar, headlines.
        /// Returns parsed report JSON. Non-fatal: individual sections may fail.
        /// </summary>
        Task<PerceptionIngestResult> RunIngestAsync(
            int lookbackDays = 365,
            int headlineLimit = 15,
            int calendarHorizonDays = 30,
            CancellationToken ct = default);

        /// <summary>
        /// Read the local perception snapshot (no network calls).
        /// </summary>
        Task<PythonRouteResult> ReadSnapshotAsync(
            int headlineLimit = 10,
            CancellationToken ct = default);
    }
}

public sealed record PerceptionIngestResult(
    bool Success,
    int ProxiesSucceeded,
    int ProxiesTotal,
    bool CalendarOk,
    string? CalendarProvider,
    int CalendarEventCount,
    bool HeadlinesOk,
    string? HeadlinesProvider,
    int HeadlineCount,
    string? ManifestPath,
    long DurationMs,
    string? ErrorMessage);

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

public sealed record MarketQuoteDto(
    string Symbol,
    double Price,
    string TimestampUtc);

public sealed record MarketCandlesDto(
    string Symbol,
    string Tf,
    JsonElement Candles,
    long? NextTo);

public sealed record MarketCandlesQuery(
    string Symbol,
    string Tf,
    string Range,
    int Limit,
    string? To);

public sealed record MarketQuoteFetchResult(
    bool Success,
    MarketQuoteDto? Quote,
    string? ErrorMessage);

public sealed record MarketCandlesFetchResult(
    bool Success,
    MarketCandlesDto? Candles,
    string? ErrorMessage);

public sealed record MarketIngestionRunResult(
    bool Success,
    IngestionReport? Report,
    string? ErrorMessage);

// ── On-Demand Perception Contracts ──────────────────────────────

/// <summary>Request for on-demand candle perception.</summary>
public sealed record MarketPerceptionRequest
{
    public required string Symbol { get; init; }
    public string Interval { get; init; } = "1d";
    public int LookbackDays { get; init; } = 90;
}

/// <summary>How the historical data was sourced.</summary>
public enum MarketHistorySource
{
    /// <summary>Data came from the local parquet data lake.</summary>
    Local,
    /// <summary>Data was freshly fetched and persisted (first time for this symbol).</summary>
    LiveBootstrap,
    /// <summary>Local data existed but was refreshed from live source.</summary>
    LiveRefresh,
    /// <summary>No historical data available.</summary>
    Missing
}

/// <summary>
/// Result of an on-demand market perception request.
/// Contains local history status, optional live quote overlay, and warnings.
/// </summary>
public sealed record MarketPerceptionResult
{
    public required bool Success { get; init; }
    public required string Symbol { get; init; }
    public required MarketHistorySource HistorySource { get; init; }

    /// <summary>Local parquet data (JSON candles) if available.</summary>
    public string? LocalDataJson { get; init; }

    /// <summary>Number of rows in local history.</summary>
    public int? LocalRowCount { get; init; }

    /// <summary>When the local data was last ingested.</summary>
    public DateTime? LocalDataAsOfUtc { get; init; }

    /// <summary>Parquet file path if data was persisted.</summary>
    public string? ParquetPath { get; init; }

    /// <summary>Live quote overlay (latest price snapshot).</summary>
    public MarketQuoteDto? QuoteOverlay { get; init; }

    /// <summary>Whether a live fetch was performed in this request.</summary>
    public bool LiveFetchOccurred { get; init; }

    /// <summary>Whether data was persisted to the data lake in this request.</summary>
    public bool DataPersisted { get; init; }

    /// <summary>Non-fatal warnings (e.g. quote overlay failed but local data available).</summary>
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();

    public string? ErrorMessage { get; init; }
}
