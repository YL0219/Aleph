using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace Aleph;

public sealed class Arbiter : IArbiter
{
    private const int MaxAgentIterations = 10;
    private const int MaxToolsPerTurn = 6;
    private const int MaxParallelToolTasks = 4;
    private const int MaxStateChangingPerTurn = 1;

    private static readonly HttpClient Client = new();
    private static readonly string ModelId = "gpt-4o-mini";

    private readonly IAxiom _axiom;
    private readonly IConfiguration _config;
    private readonly ILogger<Arbiter> _logger;

    private sealed record ToolCallInfo(int Index, string ToolCallId, string ToolName, string ArgumentsJson);

    private sealed record ToolRunPayload(
        string ThreadId,
        string ToolName,
        string ArgumentsJson,
        string ResultJson,
        long ExecutionTimeMs,
        bool IsSuccess,
        DateTime CreatedAtUtc);

    private sealed record ToolExecResult(
        int Index,
        string ToolCallId,
        string ToolName,
        string ToolContent,
        List<object> UiActionsLocal,
        ToolRunPayload RunPayload);

    public Arbiter(
        IAxiom axiom,
        IConfiguration config,
        ILogger<Arbiter> logger)
    {
        _axiom = axiom;
        _config = config;
        _logger = logger;

        var apiKey = _config["OpenAI:ApiKey"] ?? string.Empty;
        if (Client.DefaultRequestHeaders.Authorization is null && !string.IsNullOrWhiteSpace(apiKey))
        {
            Client.DefaultRequestHeaders.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", apiKey);
        }
    }

    public async Task<ArbiterHandleResult> HandleAsync(ChatRequest request, CancellationToken ct = default)
    {
        request ??= new ChatRequest();

        var userMessage = request.Message ?? string.Empty;
        var threadId = string.IsNullOrWhiteSpace(request.ThreadId)
            ? "default-user-session"
            : request.ThreadId;

        _logger.LogInformation("[Arbiter] Request received: {Message}", userMessage);

        var rawHistory = await _axiom.Chat.GetThreadMessagesAsync(threadId, ct);
        var messageHistory = new List<object>(rawHistory.Count);
        foreach (var msg in rawHistory)
        {
            messageHistory.Add(new { role = msg.Role, content = msg.Content });
        }

        var (aiResponse, uiActions, terminatedByCircuitBreaker, iterations) =
            await RunAutonomousAgentAsync(messageHistory, userMessage, threadId, ct);

        _logger.LogInformation(
            "[Arbiter] Response ready (iterations={Iterations}, circuitBreaker={CircuitBreaker})",
            iterations,
            terminatedByCircuitBreaker);

        return new ArbiterHandleResult(aiResponse, uiActions, terminatedByCircuitBreaker, iterations);
    }

    private static ToolExecResult MakeWarningResult(ToolCallInfo call, string warningMsg, string threadId)
    {
        return new ToolExecResult(
            call.Index,
            call.ToolCallId,
            call.ToolName,
            warningMsg,
            new List<object>(),
            new ToolRunPayload(
                threadId,
                call.ToolName,
                call.ArgumentsJson,
                warningMsg,
                0,
                false,
                DateTime.UtcNow));
    }

    private async Task<(string ResponseText, List<object> UiActions, bool TerminatedByCircuitBreaker, int Iterations)>
        RunAutonomousAgentAsync(List<object> history, string userMessage, string threadId, CancellationToken ct)
    {
        await _axiom.Chat.AppendMessageAsync(
            new ChatMessageWrite(threadId, "user", userMessage, null, DateTime.UtcNow),
            ct);

        history.Add(new { role = "user", content = userMessage });

        var uiActions = new List<object>();
        var toolsArray = _axiom.Mcp.GetOpenAiToolSchemas();

        var isAgentThinking = true;
        var finalAiResponse = string.Empty;
        var iteration = 0;

        while (isAgentThinking)
        {
            iteration++;
            if (iteration > MaxAgentIterations)
            {
                var warningMsg =
                    $"[Circuit Breaker] Agent terminated after {MaxAgentIterations} iterations to prevent runaway token consumption.";

                _logger.LogWarning("[Arbiter] {Warning}", warningMsg);

                await _axiom.Chat.AppendMessageAsync(
                    new ChatMessageWrite(threadId, "system", warningMsg, null, DateTime.UtcNow),
                    ct);

                var safeResponse = string.IsNullOrEmpty(finalAiResponse)
                    ? warningMsg
                    : $"{finalAiResponse}\n\n{warningMsg}";

                return (safeResponse, uiActions, true, iteration - 1);
            }

            var payload = new
            {
                model = ModelId,
                messages = history,
                temperature = 0.3,
                tools = toolsArray,
                tool_choice = "auto"
            };

            using var content = new StringContent(
                JsonSerializer.Serialize(payload),
                Encoding.UTF8,
                "application/json");

            try
            {
                var response = await Client.PostAsync(
                    "https://api.openai.com/v1/chat/completions",
                    content,
                    ct);

                var responseString = await response.Content.ReadAsStringAsync(ct);

                if (!response.IsSuccessStatusCode)
                {
                    return ($"Error: API returned {response.StatusCode}\n{responseString}", uiActions, false, iteration);
                }

                using var doc = JsonDocument.Parse(responseString);
                var choice = doc.RootElement.GetProperty("choices")[0];
                var finishReason = choice.GetProperty("finish_reason").GetString() ?? string.Empty;
                var message = choice.GetProperty("message");

                if (finishReason == "tool_calls")
                {
                    var toolCalls = message.GetProperty("tool_calls");
                    var toolCount = toolCalls.GetArrayLength();

                    _logger.LogInformation(
                        "[Arbiter] AI requested {ToolCount} tool(s) (iteration {Iteration}/{Max})",
                        toolCount,
                        iteration,
                        MaxAgentIterations);

                    history.Add(new
                    {
                        role = "assistant",
                        content = (string?)null,
                        tool_calls = toolCalls.Clone()
                    });

                    var allCalls = new List<ToolCallInfo>();
                    var idx = 0;
                    foreach (var tc in toolCalls.EnumerateArray())
                    {
                        allCalls.Add(new ToolCallInfo(
                            idx++,
                            tc.GetProperty("id").GetString() ?? string.Empty,
                            tc.GetProperty("function").GetProperty("name").GetString() ?? string.Empty,
                            tc.GetProperty("function").GetProperty("arguments").GetString() ?? string.Empty));
                    }

                    var activeCalls = allCalls.Take(MaxToolsPerTurn).ToList();
                    var overflowCalls = allCalls.Skip(MaxToolsPerTurn).ToList();

                    var readOnlyCalls = activeCalls
                        .Where(c => !_axiom.Mcp.IsStateChangingTool(c.ToolName))
                        .ToList();

                    var writeCalls = activeCalls
                        .Where(c => _axiom.Mcp.IsStateChangingTool(c.ToolName))
                        .ToList();

                    var allowedWrite = writeCalls.FirstOrDefault();
                    var rejectedWrites = writeCalls.Skip(MaxStateChangingPerTurn).ToList();

                    var allResults = new List<ToolExecResult>();

                    foreach (var oc in overflowCalls)
                    {
                        _logger.LogWarning(
                            "[Arbiter] Overflow: tool '{ToolName}' rejected (>{Max} per turn).",
                            oc.ToolName,
                            MaxToolsPerTurn);

                        allResults.Add(MakeWarningResult(
                            oc,
                            $"SYSTEM WARNING: max {MaxToolsPerTurn} tool calls per turn exceeded; this call was ignored.",
                            threadId));
                    }

                    foreach (var rw in rejectedWrites)
                    {
                        _logger.LogWarning(
                            "[Arbiter] Policy: duplicate state-changing tool '{ToolName}' rejected.",
                            rw.ToolName);

                        allResults.Add(MakeWarningResult(
                            rw,
                            "SYSTEM WARNING: multiple state-changing tools requested in one turn; only the first is allowed.",
                            threadId));
                    }

                    if (readOnlyCalls.Count > 0)
                    {
                        using var gate = new SemaphoreSlim(MaxParallelToolTasks, MaxParallelToolTasks);
                        var readTasks = readOnlyCalls.Select(async call =>
                        {
                            await gate.WaitAsync(ct);
                            try
                            {
                                return await ExecuteToolAsync(call, threadId, ct);
                            }
                            finally
                            {
                                gate.Release();
                            }
                        }).ToList();

                        var readResults = await Task.WhenAll(readTasks);
                        allResults.AddRange(readResults);
                    }

                    if (allowedWrite is not null)
                    {
                        var writeResult = await ExecuteToolAsync(allowedWrite, threadId, ct);
                        allResults.Add(writeResult);
                    }

                    allResults.Sort((a, b) => a.Index.CompareTo(b.Index));

                    foreach (var r in allResults)
                    {
                        history.Add(new
                        {
                            role = "tool",
                            tool_call_id = r.ToolCallId,
                            name = r.ToolName,
                            content = r.ToolContent
                        });

                        uiActions.AddRange(r.UiActionsLocal);
                    }

                    try
                    {
                        var runWrites = allResults
                            .Select(r => new ToolRunWrite(
                                threadId,
                                r.RunPayload.ToolName,
                                r.RunPayload.ArgumentsJson,
                                r.RunPayload.ResultJson,
                                r.RunPayload.ExecutionTimeMs,
                                r.RunPayload.IsSuccess,
                                r.RunPayload.CreatedAtUtc))
                            .ToList();

                        await _axiom.ToolRuns.AppendBatchAsync(runWrites, ct);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "[Arbiter] Failed to batch-log tool runs.");
                    }
                }
                else
                {
                    finalAiResponse = message.GetProperty("content").GetString() ?? string.Empty;
                    history.Add(new { role = "assistant", content = finalAiResponse });

                    await _axiom.Chat.AppendMessageAsync(
                        new ChatMessageWrite(threadId, "assistant", finalAiResponse, null, DateTime.UtcNow),
                        ct);

                    isAgentThinking = false;
                }
            }
            catch (Exception ex)
            {
                return ($"Connection Failed: {ex.Message}", uiActions, false, iteration);
            }
        }

        return (finalAiResponse, uiActions, false, iteration);
    }

    private async Task<ToolExecResult> ExecuteToolAsync(
        ToolCallInfo call,
        string threadId,
        CancellationToken ct)
    {
        _logger.LogInformation("[Arbiter] Executing tool: {ToolName} (idx={Index})", call.ToolName, call.Index);

        var sw = Stopwatch.StartNew();
        string toolContent;
        bool isSuccess;
        List<object> localUiActions;

        try
        {
            var invokeResult = await _axiom.Mcp.InvokeAsync(call.ToolName, call.ArgumentsJson, ct);
            toolContent = invokeResult.ToolContent;
            isSuccess = invokeResult.IsSuccess;
            localUiActions = invokeResult.UiActions.ToList();
        }
        catch (OperationCanceledException)
        {
            toolContent = "SYSTEM ERROR: Operation was cancelled.";
            isSuccess = false;
            localUiActions = new List<object>();
        }
        catch (Exception ex)
        {
            toolContent = $"SYSTEM ERROR: Exception: {ex.Message}";
            isSuccess = false;
            localUiActions = new List<object>();
        }

        sw.Stop();

        var runPayload = new ToolRunPayload(
            threadId,
            call.ToolName,
            call.ArgumentsJson,
            toolContent,
            sw.ElapsedMilliseconds,
            isSuccess,
            DateTime.UtcNow);

        _logger.LogInformation(
            "[Arbiter] Tool '{ToolName}' completed in {Ms}ms (success={Success})",
            call.ToolName,
            sw.ElapsedMilliseconds,
            isSuccess);

        return new ToolExecResult(
            call.Index,
            call.ToolCallId,
            call.ToolName,
            toolContent,
            localUiActions,
            runPayload);
    }
}
