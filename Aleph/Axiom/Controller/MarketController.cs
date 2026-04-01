// CONTRACT / INVARIANTS
// - Routes: GET /api/market/quote, GET /api/market/candles
// - Serves real-time market data for the chart web app (Unity frontend).
// - Symbols validated via SymbolValidator before any external call.
// - Market data fetches route through IAxiom market gateway.
// - Responses cached 10s via IAxiom market gateway.

using Microsoft.AspNetCore.Mvc;

namespace Aleph
{
    [ApiController]
    [Route("api/market")]
    public class MarketController : ControllerBase
    {
        private static readonly HashSet<string> AllowedTimeframes = new(StringComparer.OrdinalIgnoreCase)
            { "1m", "5m", "15m", "1h", "1d", "1w", "1mo" };

        private static readonly HashSet<string> AllowedRanges = new(StringComparer.OrdinalIgnoreCase)
            { "7d", "30d", "90d", "180d", "1y", "2y" };

        private const int DefaultLimit = 500;
        private const int MaxLimit = 2000;

        private readonly IAxiom _axiom;
        private readonly ILogger<MarketController> _logger;

        public MarketController(
            IAxiom axiom,
            ILogger<MarketController> logger)
        {
            _axiom = axiom;
            _logger = logger;
        }


        // GET /api/market/quote?symbol=SI=F
        [HttpGet("quote")]
        public async Task<IActionResult> GetQuote([FromQuery] string symbol, CancellationToken ct)
        {
            if (!SymbolValidator.TryNormalize(symbol, out var normalized))
                return BadRequest(new { error = "Invalid symbol. Must be 1-15 uppercase market-symbol characters (A-Z, 0-9, dot, hyphen, equals)." });

            _logger.LogInformation("[Market] Quote request for {Symbol}", normalized);

            var result = await _axiom.Market.GetQuoteAsync(normalized, ct);
            if (!result.Success)
            {
                return StatusCode(502, new { error = result.ErrorMessage ?? "Unknown" });
            }

            if (result.Quote is null)
            {
                return StatusCode(502, new { error = "Invalid response from data fetcher." });
            }

            return Ok(new
            {
                symbol = result.Quote.Symbol,
                price = result.Quote.Price,
                timestampUtc = result.Quote.TimestampUtc
            });
        }


        // GET /api/market/candles?symbol=SI=F&tf=1d&range=180d&limit=500&to=optional
        [HttpGet("candles")]
        public async Task<IActionResult> GetCandles(
            [FromQuery] string symbol,
            [FromQuery] string tf = "1d",
            [FromQuery] string range = "180d",
            [FromQuery] int limit = DefaultLimit,
            [FromQuery] string? to = null,
            CancellationToken ct = default)
        {
            // Validate symbol
            if (!SymbolValidator.TryNormalize(symbol, out var normalized))
                return BadRequest(new { error = "Invalid symbol." });

            tf = tf.ToLowerInvariant();
            range = range.ToLowerInvariant();

            // Validate timeframe
            if (!AllowedTimeframes.Contains(tf))
                return BadRequest(new { error = $"Invalid timeframe '{tf}'. Allowed: {string.Join(", ", AllowedTimeframes)}" });

            // Validate range
            if (!AllowedRanges.Contains(range))
                return BadRequest(new { error = $"Invalid range '{range}'. Allowed: {string.Join(", ", AllowedRanges)}" });

            // Clamp limit
            limit = Math.Clamp(limit, 1, MaxLimit);

            // Validate 'to' if provided (must be numeric unix timestamp)
            if (to != null && !double.TryParse(to, out _))
                return BadRequest(new { error = "Invalid 'to' parameter. Must be a unix timestamp." });

            _logger.LogInformation("[Market] Candles request: {Symbol} tf={Tf} range={Range} limit={Limit} to={To}",
                normalized, tf, range, limit, to ?? "latest");

            var result = await _axiom.Market.GetCandlesAsync(
                new MarketCandlesQuery(normalized, tf, range, limit, to),
                ct);
            if (!result.Success)
            {
                return StatusCode(502, new { error = result.ErrorMessage ?? "Unknown" });
            }

            if (result.Candles is null)
            {
                return StatusCode(502, new { error = "Invalid response from data fetcher." });
            }

            return Ok(new
            {
                symbol = result.Candles.Symbol,
                tf = result.Candles.Tf,
                candles = result.Candles.Candles,
                nextTo = result.Candles.NextTo
            });
        }
    }
}
