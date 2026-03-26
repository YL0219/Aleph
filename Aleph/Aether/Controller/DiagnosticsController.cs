using Microsoft.AspNetCore.Mvc;

namespace Aleph;

/// <summary>
/// Lightweight read-only diagnostics endpoint for operational observability.
///
/// Exposes homeostasis vitals, Aether cortex status, and system health
/// without any side effects. Safe to poll from monitoring dashboards.
///
/// All endpoints are GET-only, read-only, and return JSON.
/// </summary>
[ApiController]
[Route("api/diagnostics")]
public sealed class DiagnosticsController : ControllerBase
{
    private readonly IHomeostasis _homeostasis;
    private readonly IAether _aether;
    private readonly ILogger<DiagnosticsController> _logger;

    public DiagnosticsController(
        IHomeostasis homeostasis,
        IAether aether,
        ILogger<DiagnosticsController> logger)
    {
        _homeostasis = homeostasis;
        _aether = aether;
        _logger = logger;
    }

    /// <summary>
    /// Quick health check — returns homeostasis vitals and overall health assessment.
    /// No Python calls. Sub-millisecond response.
    /// </summary>
    [HttpGet("health")]
    public IActionResult GetHealth()
    {
        var snapshot = _homeostasis.GetSnapshot();

        var health = DetermineHealth(snapshot);

        return Ok(new
        {
            status = health,
            stress = Math.Round(snapshot.StressLevel, 3),
            fatigue = Math.Round(snapshot.FatigueLevel, 3),
            overload = Math.Round(snapshot.OverloadLevel, 3),
            failureStreak = snapshot.FailureStreak,
            isOverloaded = _homeostasis.IsOverloaded,
            isBreathless = _homeostasis.IsBreathless,
            lastPulseDurationMs = snapshot.LastPulseDurationMs,
            lastUpdatedUtc = snapshot.LastUpdatedUtc,
            activeFlags = snapshot.ActiveFlags,
            recentStressSources = snapshot.RecentStressSources,
            timestampUtc = DateTimeOffset.UtcNow,
        });
    }

    /// <summary>
    /// Cortex status — queries the Python ML layer for model state, pending/resolved counts.
    /// Makes one Python subprocess call. Expect ~500ms response time.
    /// </summary>
    [HttpGet("cortex")]
    public async Task<IActionResult> GetCortexStatus(
        [FromQuery] string symbol = "BTCUSDT",
        [FromQuery] string horizon = "1d",
        CancellationToken ct = default)
    {
        try
        {
            var result = await _aether.Ml.CortexStatusAsync(new MlCortexStatusRequest
            {
                Symbol = symbol,
                ActiveHorizon = horizon,
            }, ct);

            if (!result.Success)
            {
                return Ok(new
                {
                    status = "error",
                    error = result.Error,
                    timedOut = result.TimedOut,
                });
            }

            // Pass through the Python JSON directly — it's already well-structured
            return Content(result.PayloadJson, "application/json");
        }
        catch (OperationCanceledException)
        {
            return StatusCode(499, new { status = "cancelled" });
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[Diagnostics] Cortex status query failed.");
            return StatusCode(500, new { status = "error", error = ex.Message });
        }
    }

    /// <summary>
    /// Combined snapshot — homeostasis + cortex status in one call.
    /// Makes one Python call. Useful for dashboard rendering.
    /// </summary>
    [HttpGet("snapshot")]
    public async Task<IActionResult> GetSnapshot(
        [FromQuery] string symbol = "BTCUSDT",
        [FromQuery] string horizon = "1d",
        CancellationToken ct = default)
    {
        var snapshot = _homeostasis.GetSnapshot();
        var health = DetermineHealth(snapshot);

        object? cortex = null;
        try
        {
            var result = await _aether.Ml.CortexStatusAsync(new MlCortexStatusRequest
            {
                Symbol = symbol,
                ActiveHorizon = horizon,
            }, ct);

            cortex = result.Success
                ? new { ok = true, payload = result.PayloadJson }
                : new { ok = false, payload = result.Error ?? "unknown error" };
        }
        catch (Exception ex)
        {
            cortex = new { ok = false, payload = ex.Message };
        }

        return Ok(new
        {
            health,
            homeostasis = new
            {
                stress = Math.Round(snapshot.StressLevel, 3),
                fatigue = Math.Round(snapshot.FatigueLevel, 3),
                overload = Math.Round(snapshot.OverloadLevel, 3),
                failureStreak = snapshot.FailureStreak,
                isOverloaded = _homeostasis.IsOverloaded,
                isBreathless = _homeostasis.IsBreathless,
            },
            cortex,
            timestampUtc = DateTimeOffset.UtcNow,
        });
    }

    /// <summary>
    /// Operational status — rich pipeline health with maturity timeline, schema health,
    /// and training readiness. Makes one Python call. Clearly distinguishes healthy waiting
    /// from stalled/broken states.
    /// </summary>
    [HttpGet("operational")]
    public async Task<IActionResult> GetOperationalStatus(
        [FromQuery] string symbol = "BTCUSDT",
        [FromQuery] string horizon = "1d",
        [FromQuery] string interval = "1h",
        CancellationToken ct = default)
    {
        try
        {
            var result = await _aether.Ml.CortexOperationalStatusAsync(new MlCortexOperationalStatusRequest
            {
                Symbol = symbol,
                ActiveHorizon = horizon,
                Interval = interval,
            }, ct);

            if (!result.Success)
                return Ok(new { status = "error", error = result.Error, timedOut = result.TimedOut });

            return Content(result.PayloadJson, "application/json");
        }
        catch (OperationCanceledException)
        {
            return StatusCode(499, new { status = "cancelled" });
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[Diagnostics] Operational status query failed.");
            return StatusCode(500, new { status = "error", error = ex.Message });
        }
    }

    /// <summary>
    /// Dream state list — returns all simulation dreams with their status.
    /// </summary>
    [HttpGet("dreams")]
    public async Task<IActionResult> GetDreamList(CancellationToken ct = default)
    {
        try
        {
            var result = await _aether.Sim.DreamListAsync(ct);
            if (!result.Success)
                return Ok(new { status = "error", error = result.Error });
            return Content(result.PayloadJson, "application/json");
        }
        catch (OperationCanceledException)
        {
            return StatusCode(499, new { status = "cancelled" });
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[Diagnostics] Dream list query failed.");
            return StatusCode(500, new { status = "error", error = ex.Message });
        }
    }

    /// <summary>
    /// Dream state status — returns detailed status for a specific dream.
    /// </summary>
    [HttpGet("dreams/{dreamId}")]
    public async Task<IActionResult> GetDreamStatus(string dreamId, CancellationToken ct = default)
    {
        try
        {
            var result = await _aether.Sim.DreamStatusAsync(new DreamStatusRequest { DreamId = dreamId }, ct);
            if (!result.Success)
                return Ok(new { status = "error", error = result.Error });
            return Content(result.PayloadJson, "application/json");
        }
        catch (OperationCanceledException)
        {
            return StatusCode(499, new { status = "cancelled" });
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[Diagnostics] Dream status query failed for {DreamId}.", dreamId);
            return StatusCode(500, new { status = "error", error = ex.Message });
        }
    }

    private static string DetermineHealth(HomeostasisSnapshot snapshot)
    {
        if (snapshot.OverloadLevel >= 0.7 || snapshot.FailureStreak >= 3)
            return "critical";
        if (snapshot.FatigueLevel >= 0.85)
            return "critical";
        if (snapshot.StressLevel >= 0.6 || snapshot.FatigueLevel >= 0.5 || snapshot.OverloadLevel >= 0.4)
            return "degraded";
        return "healthy";
    }
}
