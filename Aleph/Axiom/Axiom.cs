using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.EntityFrameworkCore;

namespace Aleph;

public sealed class Axiom : IAxiom
{
    private static readonly Regex SkillNamePattern =
        new("^[a-z][a-z0-9_]{0,63}$", RegexOptions.Compiled);

    private readonly IDbContextFactory<AppDbContext> _dbFactory;
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly PythonDispatcherService _pythonDispatcher;
    private readonly McpToolInvoker _mcpInvoker;
    private readonly McpToolSchemaAdapter _mcpSchemaAdapter;
    private readonly ISkillRegistry _skillRegistry;
    private readonly ILogger<Axiom> _logger;

    public IAxiom.IPythonRouter Python { get; }
    public IAxiom.IMcpGateway Mcp { get; }
    public IAxiom.ITradeGateway Trades { get; }
    public IAxiom.IChatGateway Chat { get; }
    public IAxiom.IToolRunGateway ToolRuns { get; }
    public IAxiom.ISkillGateway Skills { get; }

    public Axiom(
        IDbContextFactory<AppDbContext> dbFactory,
        IServiceScopeFactory scopeFactory,
        PythonDispatcherService pythonDispatcher,
        McpToolInvoker mcpInvoker,
        McpToolSchemaAdapter mcpSchemaAdapter,
        ISkillRegistry skillRegistry,
        ILogger<Axiom> logger)
    {
        _dbFactory = dbFactory;
        _scopeFactory = scopeFactory;
        _pythonDispatcher = pythonDispatcher;
        _mcpInvoker = mcpInvoker;
        _mcpSchemaAdapter = mcpSchemaAdapter;
        _skillRegistry = skillRegistry;
        _logger = logger;

        Python = new PythonRouterGateway(this);
        Mcp = new McpGateway(this);
        Trades = new TradeGateway(this);
        Chat = new ChatGateway(this);
        ToolRuns = new ToolRunGateway(this);
        Skills = new SkillGateway(this);
    }

    private sealed class PythonRouterGateway : IAxiom.IPythonRouter
    {
        private readonly Axiom _root;

        public PythonRouterGateway(Axiom root)
        {
            _root = root;
        }

        public async Task<PythonRouteResult> RunJsonAsync(
            string domain,
            string action,
            IReadOnlyList<string> arguments,
            int timeoutMs,
            CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(domain))
                throw new ArgumentException("Domain is required.", nameof(domain));

            if (string.IsNullOrWhiteSpace(action))
                throw new ArgumentException("Action is required.", nameof(action));

            if (timeoutMs <= 0)
                throw new ArgumentOutOfRangeException(nameof(timeoutMs), "Timeout must be > 0.");

            var safeArgs = arguments ?? Array.Empty<string>();
            _root._logger.LogDebug(
                "[Axiom] Python route {Domain}/{Action} with {ArgCount} args.",
                domain,
                action,
                safeArgs.Count);
            var result = await _root._pythonDispatcher.RunAsync(domain, action, safeArgs, timeoutMs, ct);

            return new PythonRouteResult(
                result.Success,
                result.Stdout,
                result.Stderr,
                result.ExitCode,
                result.TimedOut);
        }
    }

    private sealed class McpGateway : IAxiom.IMcpGateway
    {
        private readonly Axiom _root;

        public McpGateway(Axiom root)
        {
            _root = root;
        }

        public IReadOnlyList<System.Text.Json.Nodes.JsonNode> GetOpenAiToolSchemas()
        {
            return _root._mcpSchemaAdapter.GetOpenAiToolSchemas();
        }

        public bool IsStateChangingTool(string toolName)
        {
            return _root._mcpInvoker.IsStateChangingTool(toolName);
        }

        public async Task<McpInvokeResult> InvokeAsync(
            string toolName,
            string argumentsJson,
            CancellationToken ct = default)
        {
            var result = await _root._mcpInvoker.InvokeAsync(toolName, argumentsJson, ct);
            return new McpInvokeResult(
                result.ToolContent,
                result.UiActions.ToArray(),
                result.IsSuccess,
                result.Error);
        }
    }

    private sealed class TradeGateway : IAxiom.ITradeGateway
    {
        private readonly Axiom _root;

        public TradeGateway(Axiom root)
        {
            _root = root;
        }

        public async Task<TradeExecutionResult> ExecuteTradeAsync(
            ExecuteTradeRequest request,
            CancellationToken ct = default)
        {
            if (request is null)
                throw new ArgumentNullException(nameof(request));

            if (string.IsNullOrWhiteSpace(request.ClientRequestId))
            {
                _root._logger.LogWarning("[Axiom] Trade rejected: missing ClientRequestId.");
                return new TradeExecutionResult(false, false, false, null, "ClientRequestId is required.");
            }

            if (!SymbolValidator.TryNormalize(request.Symbol, out var normalizedSymbol))
            {
                _root._logger.LogWarning("[Axiom] Trade rejected: invalid symbol '{Symbol}'.", request.Symbol);
                return new TradeExecutionResult(false, false, false, null, "Invalid symbol format.");
            }

            string side = (request.Side ?? string.Empty).Trim().ToUpperInvariant();
            if (side is not ("BUY" or "SELL"))
            {
                return new TradeExecutionResult(false, false, false, null, "Side must be BUY or SELL.");
            }

            if (request.Quantity <= 0)
            {
                return new TradeExecutionResult(false, false, false, null, "Quantity must be greater than 0.");
            }

            if (request.ExecutedPrice <= 0)
            {
                return new TradeExecutionResult(false, false, false, null, "ExecutedPrice must be greater than 0.");
            }

            ct.ThrowIfCancellationRequested();

            using var scope = _root._scopeFactory.CreateScope();
            var tradingService = scope.ServiceProvider.GetRequiredService<TradingService>();

            var tradeRequest = new TradeRequest
            {
                ClientRequestId = request.ClientRequestId,
                Symbol = normalizedSymbol,
                Side = side,
                Quantity = request.Quantity,
                ExecutedPrice = request.ExecutedPrice,
                Fees = request.Fees,
                Currency = request.Currency,
                Notes = request.Notes,
                RawJson = request.RawJson
            };

            var result = await tradingService.ExecuteTradeAsync(tradeRequest);
            return new TradeExecutionResult(
                result.Success,
                result.IsDuplicate,
                result.IsConflict,
                result.Trade is null ? null : MapTrade(result.Trade),
                result.ErrorMessage);
        }
    }

    private sealed class ChatGateway : IAxiom.IChatGateway
    {
        private static readonly HashSet<string> AllowedRoles = new(StringComparer.OrdinalIgnoreCase)
        {
            "system", "user", "assistant", "tool"
        };

        private readonly Axiom _root;

        public ChatGateway(Axiom root)
        {
            _root = root;
        }

        public async Task<IReadOnlyList<ChatMessageDto>> GetThreadMessagesAsync(
            string threadId,
            CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(threadId))
                throw new ArgumentException("ThreadId is required.", nameof(threadId));

            await using var db = await _root._dbFactory.CreateDbContextAsync(ct);
            return await db.ChatMessages
                .AsNoTracking()
                .Where(m => m.ThreadId == threadId)
                .OrderBy(m => m.CreatedAtUtc)
                .Select(m => new ChatMessageDto(
                    m.ThreadId,
                    m.Role,
                    m.Content,
                    m.MetadataJson,
                    m.CreatedAtUtc))
                .ToListAsync(ct);
        }

        public async Task AppendMessageAsync(
            ChatMessageWrite message,
            CancellationToken ct = default)
        {
            if (message is null)
                throw new ArgumentNullException(nameof(message));

            if (string.IsNullOrWhiteSpace(message.ThreadId))
                throw new ArgumentException("ThreadId is required.", nameof(message));

            if (!AllowedRoles.Contains(message.Role))
                throw new ArgumentException($"Unsupported chat role: '{message.Role}'.", nameof(message));

            await using var db = await _root._dbFactory.CreateDbContextAsync(ct);
            db.ChatMessages.Add(new ChatMessage
            {
                ThreadId = message.ThreadId,
                Role = message.Role,
                Content = message.Content ?? string.Empty,
                MetadataJson = message.MetadataJson,
                CreatedAtUtc = message.CreatedAtUtc ?? DateTime.UtcNow
            });

            await db.SaveChangesAsync(ct);
        }
    }

    private sealed class ToolRunGateway : IAxiom.IToolRunGateway
    {
        private readonly Axiom _root;

        public ToolRunGateway(Axiom root)
        {
            _root = root;
        }

        public async Task AppendBatchAsync(
            IReadOnlyList<ToolRunWrite> runs,
            CancellationToken ct = default)
        {
            if (runs is null)
                throw new ArgumentNullException(nameof(runs));

            if (runs.Count == 0)
                return;

            await using var db = await _root._dbFactory.CreateDbContextAsync(ct);
            foreach (var run in runs)
            {
                db.ToolRuns.Add(new ToolRun
                {
                    ThreadId = run.ThreadId,
                    ToolName = run.ToolName,
                    ArgumentsJson = run.ArgumentsJson,
                    ResultJson = run.ResultJson,
                    ExecutionTimeMs = run.ExecutionTimeMs,
                    IsSuccess = run.IsSuccess,
                    CreatedAtUtc = run.CreatedAtUtc
                });
            }

            await db.SaveChangesAsync(ct);
        }
    }

    private sealed class SkillGateway : IAxiom.ISkillGateway
    {
        private static readonly JsonSerializerOptions JsonOpts = new()
        {
            WriteIndented = false
        };

        private readonly Axiom _root;

        public SkillGateway(Axiom root)
        {
            _root = root;
        }

        public string GetAvailableSkills(bool includeDeprecated = false)
        {
            var snapshot = _root._skillRegistry.Snapshot;
            var skills = snapshot.Playbooks
                .Where(p => includeDeprecated || !p.Metadata.Deprecated)
                .Select(p => new Dictionary<string, object?>
                {
                    ["skill_name"] = p.Metadata.SkillName,
                    ["display_name"] = p.Metadata.DisplayName,
                    ["version"] = p.Metadata.Version,
                    ["description"] = p.Metadata.Description,
                    ["tags"] = p.Metadata.Tags,
                    ["required_tools"] = p.Metadata.RequiredTools,
                    ["deprecated"] = p.Metadata.Deprecated
                })
                .ToList();

            var response = new Dictionary<string, object?>
            {
                ["ok"] = true,
                ["schema_version"] = 1,
                ["skills"] = skills,
                ["total_count"] = skills.Count
            };

            return JsonSerializer.Serialize(response, JsonOpts);
        }

        public string ReadPlaybook(string skillName)
        {
            if (string.IsNullOrWhiteSpace(skillName) || !SkillNamePattern.IsMatch(skillName))
            {
                return BuildErrorResponse(
                    "invalid_skill_name",
                    $"Invalid skill_name: '{skillName}'. Must match pattern: ^[a-z][a-z0-9_]{{0,63}}$");
            }

            if (skillName.Contains('/') || skillName.Contains('\\') || skillName.Contains(".."))
            {
                return BuildErrorResponse("invalid_skill_name", "Skill name must not contain path characters.");
            }

            var snapshot = _root._skillRegistry.Snapshot;
            if (!snapshot.BySkillName.TryGetValue(skillName, out var playbook))
            {
                return BuildErrorResponse("skill_not_found", $"Skill '{skillName}' not found.");
            }

            var metadata = new Dictionary<string, object?>
            {
                ["skill_name"] = playbook.Metadata.SkillName,
                ["display_name"] = playbook.Metadata.DisplayName,
                ["version"] = playbook.Metadata.Version,
                ["description"] = playbook.Metadata.Description,
                ["tags"] = playbook.Metadata.Tags,
                ["required_tools"] = playbook.Metadata.RequiredTools,
                ["deprecated"] = playbook.Metadata.Deprecated,
                ["extras"] = playbook.Metadata.Extras.Count > 0
                    ? playbook.Metadata.Extras
                    : new Dictionary<string, object>()
            };

            var response = new Dictionary<string, object?>
            {
                ["ok"] = true,
                ["skill_name"] = playbook.Metadata.SkillName,
                ["metadata"] = metadata,
                ["markdown_body"] = playbook.MarkdownBody,
                ["content_hash"] = playbook.ContentHash,
                ["file_size_bytes"] = playbook.FileSizeBytes
            };

            return JsonSerializer.Serialize(response, JsonOpts);
        }

        private static string BuildErrorResponse(string error, string message)
        {
            var response = new Dictionary<string, object?>
            {
                ["ok"] = false,
                ["error"] = error,
                ["message"] = message
            };

            return JsonSerializer.Serialize(response, JsonOpts);
        }
    }

    private static TradeSnapshotDto MapTrade(Trade trade)
    {
        return new TradeSnapshotDto(
            trade.Id,
            trade.ClientRequestId,
            trade.Symbol,
            trade.Side,
            trade.Quantity,
            trade.ExecutedPrice,
            trade.Fees,
            trade.Currency,
            trade.ExecutedAtUtc,
            trade.Status,
            trade.Notes,
            trade.RawJson,
            trade.PositionId);
    }
}
