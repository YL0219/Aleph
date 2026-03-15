// CONTRACT / INVARIANTS
// - Converts MCP tool metadata into OpenAI function-calling JSON schemas.
// - Uses IMcpToolRegistry as the single source of truth (no independent reflection scan).
// - Caches results via IMemoryCache with 24h TTL (metadata is static at runtime).
// - Thread-safe: cached reads, no mutable state.

using System.Text.Json.Nodes;
using Microsoft.Extensions.Caching.Memory;

namespace Aleph;

/// <summary>
/// Bridges MCP tool metadata to OpenAI function-calling schema format.
/// Singleton — reads from IMcpToolRegistry and caches results.
/// </summary>
public sealed class McpToolSchemaAdapter
{
    private const string SchemasCacheKey = "mcp_openai_tool_schemas";
    private const string NamesCacheKey = "mcp_openai_tool_names";
    private static readonly TimeSpan CacheTtl = TimeSpan.FromHours(24);

    private readonly IMemoryCache _cache;
    private readonly IMcpToolRegistry _registry;
    private readonly ILogger<McpToolSchemaAdapter> _logger;

    public McpToolSchemaAdapter(
        IMemoryCache cache,
        IMcpToolRegistry registry,
        ILogger<McpToolSchemaAdapter> logger)
    {
        _cache = cache;
        _registry = registry;
        _logger = logger;
    }

    /// <summary>
    /// Returns the set of MCP tool names (for routing decisions in the controller).
    /// Cached — O(1) after first call.
    /// </summary>
    public IReadOnlySet<string> GetMcpToolNames()
    {
        return _cache.GetOrCreate(NamesCacheKey, entry =>
        {
            entry.SetAbsoluteExpiration(CacheTtl);

            var names = _registry.ToolNames;

            _logger.LogInformation("[McpSchema] Discovered {Count} MCP tool name(s): {Names}",
                names.Count, string.Join(", ", names));
            return names;
        })!;
    }

    /// <summary>
    /// Returns OpenAI function-calling tool schemas generated from registry descriptors.
    /// Format: [{ type: "function", function: { name, description, parameters: { type: "object", properties, required } } }]
    /// Cached — builds only once.
    /// </summary>
    public IReadOnlyList<JsonNode> GetOpenAiToolSchemas()
    {
        return _cache.GetOrCreate(SchemasCacheKey, entry =>
        {
            entry.SetAbsoluteExpiration(CacheTtl);
            return BuildSchemasFromRegistry();
        })!;
    }

    // ================================================================
    // Schema generation from registry descriptors
    // ================================================================

    private List<JsonNode> BuildSchemasFromRegistry()
    {
        var schemas = new List<JsonNode>();

        foreach (var descriptor in _registry.Tools.Values)
        {
            var properties = new JsonObject();
            var required = new JsonArray();

            foreach (var param in descriptor.Parameters)
            {
                var paramObj = new JsonObject
                {
                    ["type"] = param.JsonSchemaType
                };

                if (param.Description != null)
                    paramObj["description"] = param.Description;

                properties[param.Name] = paramObj;

                if (!param.IsOptional)
                    required.Add(JsonValue.Create(param.Name));
            }

            var functionNode = new JsonObject
            {
                ["name"] = descriptor.Name
            };
            if (descriptor.Description != null)
                functionNode["description"] = descriptor.Description;

            functionNode["parameters"] = new JsonObject
            {
                ["type"] = "object",
                ["properties"] = properties,
                ["required"] = required
            };

            var toolSchema = new JsonObject
            {
                ["type"] = "function",
                ["function"] = functionNode
            };

            schemas.Add(toolSchema);

            _logger.LogDebug("[McpSchema] Built schema for tool: {ToolName} ({ParamCount} params)",
                descriptor.Name, descriptor.Parameters.Count);
        }

        _logger.LogInformation("[McpSchema] Built {Count} OpenAI tool schema(s) from registry.", schemas.Count);
        return schemas;
    }
}
