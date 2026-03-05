using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using LifeTrader_AI.Data;
using LifeTrader_AI.Models;
using LifeTrader_AI.Infrastructure.Mcp;

namespace LifeTraderAPI.Controllers
{
    [ApiController]
    [Route("api/ai")]
    public class AiController : ControllerBase
    {
        private const int MAX_AGENT_ITERATIONS = 10;
        private const int MAX_TOOLS_PER_TURN = 6;
        private const int MAX_PARALLEL_TOOL_TASKS = 4;
        private const int MAX_STATE_CHANGING_PER_TURN = 1;

        private static readonly HttpClient client = new();
        private static readonly string modelId = "gpt-4o-mini";

        private readonly AppDbContext _db;
        private readonly IConfiguration _config;
        private readonly ILogger<AiController> _logger;
        private readonly McpToolSchemaAdapter _schemaAdapter;
        private readonly McpToolInvoker _mcpInvoker;

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

        public AiController(
            AppDbContext db,
            IConfiguration config,
            ILogger<AiController> logger,
            McpToolSchemaAdapter schemaAdapter,
            McpToolInvoker mcpInvoker)
        {
            _db = db;
            _config = config;
            _logger = logger;
            _schemaAdapter = schemaAdapter;
            _mcpInvoker = mcpInvoker;

            string apiKey = _config["OpenAI:ApiKey"] ?? "";
            if (!client.DefaultRequestHeaders.Contains("Authorization") && !string.IsNullOrEmpty(apiKey))
            {
                client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
            }
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

        [HttpPost("ask")]
        public async Task<IActionResult> AskTheAgent([FromBody] ChatRequest request, CancellationToken ct)
        {
            _logger.LogInformation("[AI] Request received: {Message}", request.Message);
            string threadId = !string.IsNullOrWhiteSpace(request.ThreadId) ? request.ThreadId : "default-user-session";

            var rawHistory = await _db.ChatMessages
                .Where(m => m.ThreadId == threadId)
                .OrderBy(m => m.CreatedAtUtc)
                .ToListAsync(ct);

            var messageHistory = new List<object>();
            foreach (var msg in rawHistory)
            {
                messageHistory.Add(new { role = msg.Role, content = msg.Content });
            }

            var (aiResponse, uiActions, terminatedByCircuitBreaker, iterations) =
                await RunAutonomousAgent(messageHistory, request.Message, threadId, ct);

            _logger.LogInformation("[AI] Response sent (iterations={Iterations}, circuitBreaker={CircuitBreaker})",
                iterations, terminatedByCircuitBreaker);

            return Ok(new
            {
                response = aiResponse,
                uiActions = uiActions,
                terminatedByCircuitBreaker = terminatedByCircuitBreaker,
                iterations = iterations
            });
        }

        private async Task<(string ResponseText, List<object> UiActions, bool TerminatedByCircuitBreaker, int Iterations)>
            RunAutonomousAgent(List<object> history, string userMessage, string threadId, CancellationToken ct)
        {
            _db.ChatMessages.Add(new ChatMessage { ThreadId = threadId, Role = "user", Content = userMessage });
            await _db.SaveChangesAsync(ct);

            history.Add(new { role = "user", content = userMessage });

            var uiActions = new List<object>();
            var toolsArray = _schemaAdapter.GetOpenAiToolSchemas();

            bool isAgentThinking = true;
            string finalAiResponse = "";
            int iteration = 0;

            while (isAgentThinking)
            {
                iteration++;
                if (iteration > MAX_AGENT_ITERATIONS)
                {
                    string warningMsg =
                        $"[Circuit Breaker] Agent terminated after {MAX_AGENT_ITERATIONS} iterations to prevent runaway token consumption.";
                    _logger.LogWarning("[AI] {Warning}", warningMsg);

                    _db.ChatMessages.Add(new ChatMessage
                    {
                        ThreadId = threadId,
                        Role = "system",
                        Content = warningMsg
                    });
                    await _db.SaveChangesAsync(ct);

                    string safeResponse = string.IsNullOrEmpty(finalAiResponse)
                        ? warningMsg
                        : $"{finalAiResponse}\n\n{warningMsg}";

                    return (safeResponse, uiActions, true, iteration - 1);
                }

                var payload = new
                {
                    model = modelId,
                    messages = history,
                    temperature = 0.3,
                    tools = toolsArray,
                    tool_choice = "auto"
                };

                var content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");

                try
                {
                    var response = await client.PostAsync("https://api.openai.com/v1/chat/completions", content, ct);
                    string responseString = await response.Content.ReadAsStringAsync(ct);

                    if (!response.IsSuccessStatusCode)
                        return ($"Error: API returned {response.StatusCode}\n{responseString}", uiActions, false, iteration);

                    using var doc = JsonDocument.Parse(responseString);
                    var choice = doc.RootElement.GetProperty("choices")[0];
                    string finishReason = choice.GetProperty("finish_reason").GetString() ?? "";
                    var message = choice.GetProperty("message");

                    if (finishReason == "tool_calls")
                    {
                        var toolCalls = message.GetProperty("tool_calls");
                        int toolCount = toolCalls.GetArrayLength();

                        _logger.LogInformation("[AI] AI requested {ToolCount} tool(s) (iteration {Iter}/{Max})",
                            toolCount, iteration, MAX_AGENT_ITERATIONS);

                        history.Add(new
                        {
                            role = "assistant",
                            content = (string?)null,
                            tool_calls = toolCalls.Clone()
                        });

                        var allCalls = new List<ToolCallInfo>();
                        int idx = 0;
                        foreach (var tc in toolCalls.EnumerateArray())
                        {
                            allCalls.Add(new ToolCallInfo(
                                idx++,
                                tc.GetProperty("id").GetString() ?? "",
                                tc.GetProperty("function").GetProperty("name").GetString() ?? "",
                                tc.GetProperty("function").GetProperty("arguments").GetString() ?? ""));
                        }

                        var activeCalls = allCalls.Take(MAX_TOOLS_PER_TURN).ToList();
                        var overflowCalls = allCalls.Skip(MAX_TOOLS_PER_TURN).ToList();

                        var readOnlyCalls = activeCalls
                            .Where(c => !_mcpInvoker.IsStateChangingTool(c.ToolName))
                            .ToList();
                        var writeCalls = activeCalls
                            .Where(c => _mcpInvoker.IsStateChangingTool(c.ToolName))
                            .ToList();

                        ToolCallInfo? allowedWrite = writeCalls.FirstOrDefault();
                        var rejectedWrites = writeCalls.Skip(MAX_STATE_CHANGING_PER_TURN).ToList();

                        var allResults = new List<ToolExecResult>();

                        foreach (var oc in overflowCalls)
                        {
                            _logger.LogWarning("[AI] Overflow: tool '{ToolName}' rejected (>{Max} per turn).",
                                oc.ToolName, MAX_TOOLS_PER_TURN);

                            allResults.Add(MakeWarningResult(
                                oc,
                                $"SYSTEM WARNING: max {MAX_TOOLS_PER_TURN} tool calls per turn exceeded; this call was ignored.",
                                threadId));
                        }

                        foreach (var rw in rejectedWrites)
                        {
                            _logger.LogWarning("[AI] Policy: duplicate state-changing tool '{ToolName}' rejected.", rw.ToolName);

                            allResults.Add(MakeWarningResult(
                                rw,
                                "SYSTEM WARNING: multiple state-changing tools requested in one turn; only the first is allowed.",
                                threadId));
                        }

                        if (readOnlyCalls.Count > 0)
                        {
                            using var gate = new SemaphoreSlim(MAX_PARALLEL_TOOL_TASKS, MAX_PARALLEL_TOOL_TASKS);

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

                        if (allowedWrite != null)
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
                            foreach (var r in allResults)
                            {
                                var p = r.RunPayload;
                                _db.ToolRuns.Add(new ToolRun
                                {
                                    ThreadId = p.ThreadId,
                                    ToolName = p.ToolName,
                                    ArgumentsJson = p.ArgumentsJson,
                                    ResultJson = p.ResultJson,
                                    ExecutionTimeMs = p.ExecutionTimeMs,
                                    IsSuccess = p.IsSuccess,
                                    CreatedAtUtc = p.CreatedAtUtc
                                });
                            }
                            await _db.SaveChangesAsync(ct);
                        }
                        catch (Exception logEx)
                        {
                            _logger.LogError(logEx, "[AI] Failed to batch-log ToolRuns.");
                        }
                    }
                    else
                    {
                        finalAiResponse = message.GetProperty("content").GetString() ?? "";
                        history.Add(new { role = "assistant", content = finalAiResponse });

                        _db.ChatMessages.Add(new ChatMessage
                        {
                            ThreadId = threadId,
                            Role = "assistant",
                            Content = finalAiResponse
                        });
                        await _db.SaveChangesAsync(ct);

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
            _logger.LogInformation("[AI] Executing tool: {ToolName} (idx={Index})", call.ToolName, call.Index);

            var sw = Stopwatch.StartNew();
            string toolContent;
            bool isSuccess;
            List<object> localUiActions;

            try
            {
                var invokeResult = await _mcpInvoker.InvokeAsync(call.ToolName, call.ArgumentsJson, ct);
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

            _logger.LogInformation("[AI] Tool '{ToolName}' completed in {Ms}ms (success={Success})",
                call.ToolName, sw.ElapsedMilliseconds, isSuccess);

            return new ToolExecResult(
                call.Index,
                call.ToolCallId,
                call.ToolName,
                toolContent,
                localUiActions,
                runPayload);
        }
    }
}
