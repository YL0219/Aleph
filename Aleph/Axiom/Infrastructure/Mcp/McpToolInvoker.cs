using System.Globalization;
using System.Text.Json;

namespace Aleph;

/// <summary>
/// Single execution gateway for all MCP-backed AI tool calls.
/// Uses IMcpToolRegistry for discovery — no hardcoded routing dictionary.
/// Resolves tool instances through DI and binds JSON arguments to method parameters by name.
/// </summary>
public sealed class McpToolInvoker
{
    private readonly IServiceProvider _serviceProvider;
    private readonly IMcpToolRegistry _registry;
    private readonly ILogger<McpToolInvoker> _logger;

    public McpToolInvoker(
        IServiceProvider serviceProvider,
        IMcpToolRegistry registry,
        ILogger<McpToolInvoker> logger)
    {
        _serviceProvider = serviceProvider;
        _registry = registry;
        _logger = logger;
    }

    public bool IsStateChangingTool(string toolName) => _registry.IsStateChangingTool(toolName);

    public async Task<McpToolResult> InvokeAsync(
        string toolName,
        string argumentsJson,
        CancellationToken ct = default)
    {
        _logger.LogDebug("[McpInvoker] Invoking tool: {ToolName}", toolName);

        if (!_registry.TryGetTool(toolName, out var descriptor))
        {
            string msg = $"Unknown MCP tool: '{toolName}'.";
            _logger.LogWarning("[McpInvoker] {Message}", msg);
            return BuildInvokerFailure(msg);
        }

        try
        {
            string normalizedJson = string.IsNullOrWhiteSpace(argumentsJson) ? "{}" : argumentsJson;
            using var doc = JsonDocument.Parse(normalizedJson);
            var root = doc.RootElement;

            // Resolve the tool instance from DI
            var toolInstance = _serviceProvider.GetRequiredService(descriptor.DeclaringType);

            // Bind parameters
            var bindResult = BindParameters(descriptor, root, ct);
            if (!bindResult.Success)
            {
                return BuildInvokerFailure(bindResult.Error!);
            }

            // Invoke the reflected method
            var rawResult = descriptor.Method.Invoke(toolInstance, bindResult.Arguments);

            // Normalize output based on return kind
            return await NormalizeResultAsync(descriptor, rawResult);
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex, "[McpInvoker] Invalid args JSON for tool '{ToolName}'", toolName);
            return BuildInvokerFailure($"Invalid arguments JSON for tool '{toolName}'.");
        }
        catch (System.Reflection.TargetInvocationException ex) when (ex.InnerException is not null)
        {
            _logger.LogError(ex.InnerException, "[McpInvoker] Unhandled exception invoking tool '{ToolName}'", toolName);
            return BuildInvokerFailure($"Internal error invoking tool '{toolName}'.");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[McpInvoker] Unhandled exception invoking tool '{ToolName}'", toolName);
            return BuildInvokerFailure($"Internal error invoking tool '{toolName}'.");
        }
    }

    // ─── Parameter Binding ───────────────────────────────────────────

    private ParameterBindResult BindParameters(
        McpToolDescriptor descriptor,
        JsonElement root,
        CancellationToken ct)
    {
        var method = descriptor.Method;
        var methodParams = method.GetParameters();
        var args = new object?[methodParams.Length];

        // Build a lookup from the descriptor's bindable parameters
        var bindableByName = new Dictionary<string, McpParameterDescriptor>(StringComparer.OrdinalIgnoreCase);
        foreach (var p in descriptor.Parameters)
        {
            bindableByName[p.Name] = p;
        }

        for (int i = 0; i < methodParams.Length; i++)
        {
            var mp = methodParams[i];

            // Framework-injected: CancellationToken
            if (mp.ParameterType == typeof(CancellationToken))
            {
                args[i] = ct;
                continue;
            }

            // Framework-injected: skip other framework types
            if (mp.ParameterType == typeof(IServiceProvider)
                || (mp.ParameterType.FullName?.StartsWith("ModelContextProtocol.") ?? false))
            {
                args[i] = mp.HasDefaultValue ? mp.DefaultValue : null;
                continue;
            }

            // Bindable parameter — look it up in JSON args
            string paramName = mp.Name!;

            if (root.TryGetProperty(paramName, out var jsonProp)
                && jsonProp.ValueKind != JsonValueKind.Null
                && jsonProp.ValueKind != JsonValueKind.Undefined)
            {
                if (!TryConvertJsonValue(jsonProp, mp.ParameterType, out var converted, out var convError))
                {
                    return ParameterBindResult.Fail($"Argument '{paramName}' {convError}");
                }
                args[i] = converted;
            }
            else if (mp.HasDefaultValue)
            {
                args[i] = mp.DefaultValue;
            }
            else
            {
                return ParameterBindResult.Fail($"Missing required argument '{paramName}'.");
            }
        }

        return ParameterBindResult.Ok(args);
    }

    private static bool TryConvertJsonValue(
        JsonElement el,
        Type targetType,
        out object? value,
        out string? error)
    {
        value = null;
        error = null;

        if (targetType == typeof(string))
        {
            if (el.ValueKind == JsonValueKind.String)
            {
                value = el.GetString();
                return true;
            }
            // Accept non-string JSON values as their string representation
            value = el.GetRawText().Trim('"');
            return true;
        }

        if (targetType == typeof(int))
        {
            if (TryReadInt(el, out int intVal))
            {
                value = intVal;
                return true;
            }
            error = "must be an integer.";
            return false;
        }

        if (targetType == typeof(long))
        {
            if (el.ValueKind == JsonValueKind.Number && el.TryGetInt64(out long longVal))
            {
                value = longVal;
                return true;
            }
            error = "must be a long integer.";
            return false;
        }

        if (targetType == typeof(decimal))
        {
            if (TryReadDecimal(el, out decimal decVal))
            {
                value = decVal;
                return true;
            }
            error = "must be a number.";
            return false;
        }

        if (targetType == typeof(double))
        {
            if (el.ValueKind == JsonValueKind.Number && el.TryGetDouble(out double dblVal))
            {
                value = dblVal;
                return true;
            }
            error = "must be a number.";
            return false;
        }

        if (targetType == typeof(float))
        {
            if (el.ValueKind == JsonValueKind.Number && el.TryGetSingle(out float fltVal))
            {
                value = fltVal;
                return true;
            }
            error = "must be a number.";
            return false;
        }

        if (targetType == typeof(bool))
        {
            if (el.ValueKind == JsonValueKind.True)
            {
                value = true;
                return true;
            }
            if (el.ValueKind == JsonValueKind.False)
            {
                value = false;
                return true;
            }
            error = "must be a boolean.";
            return false;
        }

        // Fallback: try to deserialize as string
        if (el.ValueKind == JsonValueKind.String)
        {
            value = el.GetString();
            return true;
        }

        error = $"cannot convert JSON {el.ValueKind} to {targetType.Name}.";
        return false;
    }

    // ─── Result Normalization ────────────────────────────────────────

    private async Task<McpToolResult> NormalizeResultAsync(
        McpToolDescriptor descriptor,
        object? rawResult)
    {
        switch (descriptor.ReturnKind)
        {
            case McpReturnKind.String:
            {
                var content = rawResult as string ?? "";
                return InferSuccess(content)
                    ? McpToolResult.Success(content)
                    : McpToolResult.Failure(content);
            }

            case McpReturnKind.TaskOfString:
            {
                var task = (Task<string>)rawResult!;
                var content = await task;
                return InferSuccess(content)
                    ? McpToolResult.Success(content)
                    : McpToolResult.Failure(content);
            }

            case McpReturnKind.TaskOfMcpToolResult:
            {
                var task = (Task<McpToolResult>)rawResult!;
                return await task;
            }

            default:
                return BuildInvokerFailure($"Unsupported return kind: {descriptor.ReturnKind}");
        }
    }

    // ─── Shared Helpers (preserved from original) ────────────────────

    private static bool TryReadInt(JsonElement el, out int value)
    {
        value = default;
        return el.ValueKind switch
        {
            JsonValueKind.Number => el.TryGetInt32(out value),
            JsonValueKind.String => int.TryParse(
                el.GetString(),
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out value),
            _ => false
        };
    }

    private static bool TryReadDecimal(JsonElement el, out decimal value)
    {
        value = default;
        return el.ValueKind switch
        {
            JsonValueKind.Number => el.TryGetDecimal(out value),
            JsonValueKind.String => decimal.TryParse(
                el.GetString(),
                NumberStyles.Number,
                CultureInfo.InvariantCulture,
                out value),
            _ => false
        };
    }

    private static bool InferSuccess(string content)
    {
        if (string.IsNullOrWhiteSpace(content))
            return false;

        if (content.StartsWith("ERROR", StringComparison.OrdinalIgnoreCase) ||
            content.StartsWith("SYSTEM ERROR", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        try
        {
            using var doc = JsonDocument.Parse(content);
            if (doc.RootElement.ValueKind == JsonValueKind.Object &&
                doc.RootElement.TryGetProperty("ok", out var okProp) &&
                okProp.ValueKind == JsonValueKind.False)
            {
                return false;
            }
        }
        catch
        {
            // Non-JSON output is valid.
        }

        return true;
    }

    private static McpToolResult BuildInvokerFailure(string message)
    {
        return McpToolResult.Failure(BuildErrorJson(message), message);
    }

    private static string BuildErrorJson(string message)
    {
        string escaped = message.Replace("\\", "\\\\").Replace("\"", "\\\"");
        return $"{{\"ok\":false,\"error\":\"{escaped}\"}}";
    }

    // ─── Internal bind result ────────────────────────────────────────

    private readonly struct ParameterBindResult
    {
        public bool Success { get; }
        public object?[]? Arguments { get; }
        public string? Error { get; }

        private ParameterBindResult(bool success, object?[]? args, string? error)
        {
            Success = success;
            Arguments = args;
            Error = error;
        }

        public static ParameterBindResult Ok(object?[] args) => new(true, args, null);
        public static ParameterBindResult Fail(string error) => new(false, null, error);
    }
}
