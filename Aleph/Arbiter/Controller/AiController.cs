using Microsoft.AspNetCore.Mvc;

namespace Aleph;

[ApiController]
[Route("api/ai")]
public sealed class AiController : ControllerBase
{
    private readonly IArbiter _arbiter;
    private readonly ILogger<AiController> _logger;

    public AiController(IArbiter arbiter, ILogger<AiController> logger)
    {
        _arbiter = arbiter;
        _logger = logger;
    }

    [HttpPost("ask")]
    public async Task<IActionResult> AskTheAgent([FromBody] ChatRequest request, CancellationToken ct)
    {
        var safeRequest = request ?? new ChatRequest();

        var result = await _arbiter.HandleAsync(safeRequest, ct);

        return Ok(new
        {
            response = result.Response,
            uiActions = result.UiActions,
            terminatedByCircuitBreaker = result.TerminatedByCircuitBreaker,
            iterations = result.Iterations
        });
    }
}
