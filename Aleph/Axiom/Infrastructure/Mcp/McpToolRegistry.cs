using System.ComponentModel;
using System.Reflection;
using ModelContextProtocol.Server;

namespace Aleph;

/// <summary>
/// Reflection-based MCP tool registry. Scans the main Aleph assembly once at construction,
/// discovers all [McpServerTool] methods on [McpServerToolType] classes, validates them,
/// and caches immutable descriptors.
///
/// Singleton — constructed once at startup.
/// Throws on duplicate public tool names (fail-fast safety).
/// Skips and logs methods with invalid signatures.
/// </summary>
public sealed class McpToolRegistry : IMcpToolRegistry
{
    private static readonly HashSet<Type> FrameworkParameterTypes = new()
    {
        typeof(CancellationToken),
        typeof(IServiceProvider)
    };

    private readonly IReadOnlyDictionary<string, McpToolDescriptor> _tools;
    private readonly IReadOnlySet<string> _toolNames;
    private readonly IReadOnlySet<string> _stateChangingTools;

    public IReadOnlyDictionary<string, McpToolDescriptor> Tools => _tools;
    public IReadOnlySet<string> ToolNames => _toolNames;

    public McpToolRegistry(ILogger<McpToolRegistry> logger)
    {
        var assembly = Assembly.GetExecutingAssembly();
        var tools = new Dictionary<string, McpToolDescriptor>(StringComparer.OrdinalIgnoreCase);

        foreach (var type in assembly.GetTypes())
        {
            if (type.GetCustomAttribute<McpServerToolTypeAttribute>() == null)
                continue;

            foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static))
            {
                var toolAttr = method.GetCustomAttribute<McpServerToolAttribute>();
                if (toolAttr == null) continue;

                string toolName = toolAttr.Name ?? method.Name;

                // Validate return type
                if (!TryClassifyReturnType(method.ReturnType, out var returnKind))
                {
                    logger.LogWarning(
                        "[McpRegistry] SKIPPING tool '{ToolName}' on {Type}.{Method}: " +
                        "unsupported return type '{ReturnType}'. Expected string, Task<string>, or Task<McpToolResult>.",
                        toolName, type.Name, method.Name, method.ReturnType.Name);
                    continue;
                }

                // Validate declaring type is not abstract/static (must be DI-resolvable)
                if (type.IsAbstract || type.IsInterface)
                {
                    logger.LogWarning(
                        "[McpRegistry] SKIPPING tool '{ToolName}': declaring type {Type} is abstract/interface and cannot be DI-resolved.",
                        toolName, type.Name);
                    continue;
                }

                // Build parameter descriptors (skip framework-injected params)
                var parameters = new List<McpParameterDescriptor>();
                foreach (var param in method.GetParameters())
                {
                    if (IsFrameworkParameter(param)) continue;

                    parameters.Add(new McpParameterDescriptor
                    {
                        Name = param.Name!,
                        ClrType = param.ParameterType,
                        IsOptional = param.HasDefaultValue,
                        DefaultValue = param.HasDefaultValue ? param.DefaultValue : null,
                        Description = param.GetCustomAttribute<DescriptionAttribute>()?.Description,
                        JsonSchemaType = MapClrTypeToJsonSchema(param.ParameterType)
                    });
                }

                // Determine state-changing from ReadOnly property
                bool isStateChanging = !toolAttr.ReadOnly;

                var descriptor = new McpToolDescriptor
                {
                    Name = toolName,
                    Description = method.GetCustomAttribute<DescriptionAttribute>()?.Description,
                    DeclaringType = type,
                    Method = method,
                    Parameters = parameters,
                    IsStateChanging = isStateChanging,
                    ReturnKind = returnKind
                };

                // Duplicate check — FAIL STARTUP
                if (tools.ContainsKey(toolName))
                {
                    var existing = tools[toolName];
                    throw new InvalidOperationException(
                        $"[McpRegistry] DUPLICATE MCP tool name '{toolName}' detected. " +
                        $"First: {existing.DeclaringType.Name}.{existing.Method.Name}, " +
                        $"Second: {type.Name}.{method.Name}. " +
                        $"Each tool must have a globally unique name.");
                }

                tools[toolName] = descriptor;

                logger.LogDebug(
                    "[McpRegistry] Registered tool '{ToolName}' -> {Type}.{Method} " +
                    "({ParamCount} params, stateChanging={StateChanging}, returns={ReturnKind})",
                    toolName, type.Name, method.Name, parameters.Count, isStateChanging, returnKind);
            }
        }

        _tools = tools;
        _toolNames = tools.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase);
        _stateChangingTools = tools
            .Where(kv => kv.Value.IsStateChanging)
            .Select(kv => kv.Key)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        logger.LogInformation(
            "[McpRegistry] Discovery complete: {Count} tool(s) registered: {Names}",
            tools.Count, string.Join(", ", tools.Keys.OrderBy(n => n)));
    }

    public bool TryGetTool(string toolName, out McpToolDescriptor descriptor)
    {
        descriptor = null!;
        if (string.IsNullOrWhiteSpace(toolName)) return false;
        return _tools.TryGetValue(toolName, out descriptor!);
    }

    public bool IsStateChangingTool(string toolName) => _stateChangingTools.Contains(toolName);

    // ─── Return Type Classification ──────────────────────────────────

    private static bool TryClassifyReturnType(Type returnType, out McpReturnKind kind)
    {
        kind = default;

        if (returnType == typeof(string))
        {
            kind = McpReturnKind.String;
            return true;
        }

        if (returnType == typeof(Task<string>))
        {
            kind = McpReturnKind.TaskOfString;
            return true;
        }

        if (returnType == typeof(Task<McpToolResult>))
        {
            kind = McpReturnKind.TaskOfMcpToolResult;
            return true;
        }

        return false;
    }

    // ─── Parameter Helpers ───────────────────────────────────────────

    private static bool IsFrameworkParameter(ParameterInfo param)
    {
        var type = param.ParameterType;
        return FrameworkParameterTypes.Contains(type)
            || (type.FullName?.StartsWith("ModelContextProtocol.") ?? false);
    }

    private static string MapClrTypeToJsonSchema(Type type)
    {
        if (type == typeof(string)) return "string";
        if (type == typeof(int) || type == typeof(long) || type == typeof(short) || type == typeof(byte)) return "integer";
        if (type == typeof(double) || type == typeof(float) || type == typeof(decimal)) return "number";
        if (type == typeof(bool)) return "boolean";
        return "string"; // safe fallback
    }
}
