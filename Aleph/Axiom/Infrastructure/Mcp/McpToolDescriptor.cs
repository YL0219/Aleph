using System.Reflection;

namespace Aleph;

/// <summary>
/// Immutable descriptor for a single MCP tool method discovered via reflection.
/// Serves as the shared source of truth for both invocation and schema generation.
/// </summary>
public sealed class McpToolDescriptor
{
    /// <summary>Public tool name exposed to the AI (from [McpServerTool(Name = ...)]).</summary>
    public required string Name { get; init; }

    /// <summary>Human-readable description (from [Description] attribute).</summary>
    public string? Description { get; init; }

    /// <summary>The CLR type that declares this tool method (resolved via DI at invocation time).</summary>
    public required Type DeclaringType { get; init; }

    /// <summary>The reflected MethodInfo for invocation.</summary>
    public required MethodInfo Method { get; init; }

    /// <summary>Parameters that should be bound from the JSON arguments (excludes framework-injected params).</summary>
    public required IReadOnlyList<McpParameterDescriptor> Parameters { get; init; }

    /// <summary>True if the tool mutates state (ReadOnly = false on the attribute).</summary>
    public required bool IsStateChanging { get; init; }

    /// <summary>How the method's return value should be handled.</summary>
    public required McpReturnKind ReturnKind { get; init; }
}

/// <summary>
/// Descriptor for a single bindable parameter on an MCP tool method.
/// </summary>
public sealed class McpParameterDescriptor
{
    public required string Name { get; init; }
    public required Type ClrType { get; init; }
    public required bool IsOptional { get; init; }
    public object? DefaultValue { get; init; }
    public string? Description { get; init; }

    /// <summary>JSON Schema type string for schema generation.</summary>
    public required string JsonSchemaType { get; init; }
}

/// <summary>
/// Classifies the return type of a discovered MCP tool method.
/// Designed to be extended (e.g. McpToolResult, Task&lt;McpToolResult&gt;) without a major refactor.
/// </summary>
public enum McpReturnKind
{
    /// <summary>Method returns string synchronously.</summary>
    String,

    /// <summary>Method returns Task&lt;string&gt;.</summary>
    TaskOfString,

    /// <summary>Method returns Task&lt;McpToolResult&gt; (internal shortcut for execute_trade/open_chart).</summary>
    TaskOfMcpToolResult
}
