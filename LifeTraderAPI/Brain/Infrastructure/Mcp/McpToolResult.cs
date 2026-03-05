namespace LifeTrader_AI.Infrastructure.Mcp
{
    /// <summary>
    /// Structured result returned by McpToolInvoker so AiController can remain tool-agnostic.
    /// </summary>
    public sealed class McpToolResult
    {
        public string ToolContent { get; init; } = "";
        public IReadOnlyList<object> UiActions { get; init; } = Array.Empty<object>();
        public bool IsSuccess { get; init; }
        public string? Error { get; init; }

        public static McpToolResult Success(
            string toolContent,
            IReadOnlyList<object>? uiActions = null)
        {
            return new McpToolResult
            {
                ToolContent = toolContent,
                UiActions = uiActions ?? Array.Empty<object>(),
                IsSuccess = true
            };
        }

        public static McpToolResult Failure(
            string toolContent,
            string? error = null,
            IReadOnlyList<object>? uiActions = null)
        {
            return new McpToolResult
            {
                ToolContent = toolContent,
                UiActions = uiActions ?? Array.Empty<object>(),
                IsSuccess = false,
                Error = error ?? toolContent
            };
        }
    }
}
