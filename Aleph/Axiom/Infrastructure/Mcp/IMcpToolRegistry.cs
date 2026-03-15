namespace Aleph;

/// <summary>
/// Shared source of truth for all MCP tool metadata.
/// Discovered once via reflection, immutable thereafter.
/// Used by both McpToolInvoker (invocation) and McpToolSchemaAdapter (schema generation).
/// </summary>
public interface IMcpToolRegistry
{
    /// <summary>All discovered tool descriptors, keyed by public tool name (case-insensitive).</summary>
    IReadOnlyDictionary<string, McpToolDescriptor> Tools { get; }

    /// <summary>Try to get a descriptor by public tool name.</summary>
    bool TryGetTool(string toolName, out McpToolDescriptor descriptor);

    /// <summary>Returns true if the tool is state-changing (ReadOnly = false).</summary>
    bool IsStateChangingTool(string toolName);

    /// <summary>All tool names (for schema adapter's name set).</summary>
    IReadOnlySet<string> ToolNames { get; }
}
