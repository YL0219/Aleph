using System.ComponentModel;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using LifeTrader_AI.Infrastructure.Python;
using ModelContextProtocol.Server;

namespace LifeTrader_AI.Infrastructure.Mcp
{
    /// <summary>
    /// MCP tools for atomic news retrieval and website scraping.
    /// All Python execution is routed through PythonDispatcherService + python_router.py.
    /// </summary>
    [McpServerToolType]
    public sealed class McpNewsTools
    {
        private const int HeadlinesDefaultLimit = 10;
        private const int HeadlinesMinLimit = 1;
        private const int HeadlinesMaxLimit = 30;
        private const int HeadlinesTimeoutMs = 25_000;

        private const int ScrapeDefaultTimeoutSec = 12;
        private const int ScrapeMinTimeoutSec = 1;
        private const int ScrapeMaxTimeoutSec = 30;
        private const int ScrapeTimeoutBufferMs = 1_500;

        private readonly PythonDispatcherService _dispatcher;
        private readonly ILogger<McpNewsTools> _logger;

        public McpNewsTools(
            PythonDispatcherService dispatcher,
            ILogger<McpNewsTools> logger)
        {
            _dispatcher = dispatcher;
            _logger = logger;
        }

        [McpServerTool(Name = "get_news_headlines", ReadOnly = true)]
        [Description(
            "Fetches recent financial news headlines. " +
            "If symbol is provided, focuses on that ticker. " +
            "Returns JSON with provider, normalized headline items, and optional error.")]
        public async Task<string> GetNewsHeadlines(
            [Description("Optional stock ticker symbol (e.g. AMD). Empty means macro/general headlines.")]
            string symbol = "",
            [Description("Maximum headlines to return (1-30). Defaults to 10.")]
            int limit = HeadlinesDefaultLimit,
            CancellationToken ct = default)
        {
            string normalizedSymbol = "";
            if (!string.IsNullOrWhiteSpace(symbol))
            {
                if (!SymbolValidator.TryNormalize(symbol, out normalizedSymbol))
                {
                    return BuildErrorJson($"Invalid symbol: '{symbol}'.");
                }
            }

            int clampedLimit = Math.Clamp(limit, HeadlinesMinLimit, HeadlinesMaxLimit);
            if (clampedLimit != limit)
            {
                _logger.LogDebug("[MCP/News] limit clamped from {Requested} to {Clamped}", limit, clampedLimit);
            }

            if (!_dispatcher.IsAvailable)
            {
                return BuildErrorJson("Python environment not available. Run setup_venv.ps1 to create the venv.");
            }

            var args = new List<string>();
            if (!string.IsNullOrWhiteSpace(normalizedSymbol))
            {
                args.Add("--symbol");
                args.Add(normalizedSymbol);
            }

            args.Add("--limit");
            args.Add(clampedLimit.ToString(CultureInfo.InvariantCulture));

            try
            {
                var result = await _dispatcher.RunAsync("news", "headlines", args, HeadlinesTimeoutMs, ct);

                if (result.TimedOut)
                {
                    return BuildErrorJson("News headlines request timed out.");
                }

                if (!result.Success)
                {
                    _logger.LogWarning(
                        "[MCP/News] headlines failed: exit={ExitCode}, stderr={Stderr}",
                        result.ExitCode, result.Stderr);

                    return BuildErrorJson($"News headlines worker failed (exit code {result.ExitCode}).");
                }

                if (string.IsNullOrWhiteSpace(result.Stdout))
                {
                    return BuildErrorJson("News headlines worker returned empty stdout.");
                }

                return result.Stdout;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MCP/News] headlines invocation failed.");
                return BuildErrorJson($"News headlines invocation failed: {ex.Message}");
            }
        }

        [McpServerTool(Name = "scrape_website_text", ReadOnly = true)]
        [Description(
            "Scrapes article/body text from a public website URL and returns JSON. " +
            "Enforces SSRF protections in C# and Python before extraction.")]
        public async Task<string> ScrapeWebsiteText(
            [Description("Target website URL. Must be publicly routable http/https.")]
            string url,
            [Description("Scrape timeout in seconds (1-30). Defaults to 12.")]
            int timeoutSec = ScrapeDefaultTimeoutSec,
            CancellationToken ct = default)
        {
            if (string.IsNullOrWhiteSpace(url))
            {
                return BuildErrorJson("URL is required.");
            }

            if (!Uri.TryCreate(url, UriKind.Absolute, out var uri))
            {
                return BuildErrorJson("Invalid URL format.");
            }

            if (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps)
            {
                return BuildErrorJson("Only http/https URLs are allowed.");
            }

            var validation = await ValidatePublicUrlAsync(uri, ct);
            if (!validation.IsAllowed)
            {
                return BuildErrorJson(validation.ErrorMessage);
            }

            if (!_dispatcher.IsAvailable)
            {
                return BuildErrorJson("Python environment not available. Run setup_venv.ps1 to create the venv.");
            }

            int clampedTimeoutSec = Math.Clamp(timeoutSec, ScrapeMinTimeoutSec, ScrapeMaxTimeoutSec);
            int timeoutMs = (clampedTimeoutSec * 1000) + ScrapeTimeoutBufferMs;

            var args = new List<string>
            {
                "--url", uri.AbsoluteUri,
                "--timeoutSec", clampedTimeoutSec.ToString(CultureInfo.InvariantCulture)
            };

            try
            {
                var result = await _dispatcher.RunAsync("news", "scrape", args, timeoutMs, ct);

                if (result.TimedOut)
                {
                    return BuildErrorJson("Website scrape timed out.");
                }

                if (!result.Success)
                {
                    _logger.LogWarning(
                        "[MCP/News] scrape failed: exit={ExitCode}, stderr={Stderr}",
                        result.ExitCode, result.Stderr);

                    return BuildErrorJson($"Website scrape worker failed (exit code {result.ExitCode}).");
                }

                if (string.IsNullOrWhiteSpace(result.Stdout))
                {
                    return BuildErrorJson("Website scrape worker returned empty stdout.");
                }

                return result.Stdout;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MCP/News] scrape invocation failed.");
                return BuildErrorJson($"Website scrape invocation failed: {ex.Message}");
            }
        }

        private async Task<(bool IsAllowed, string ErrorMessage)> ValidatePublicUrlAsync(Uri uri, CancellationToken ct)
        {
            string host = uri.DnsSafeHost;
            if (string.IsNullOrWhiteSpace(host))
            {
                return (false, "URL host is missing.");
            }

            if (IsLocalhostHost(host))
            {
                return (false, "Localhost is not allowed.");
            }

            IPAddress[] addresses;
            try
            {
                if (IPAddress.TryParse(host, out var literal))
                {
                    addresses = new[] { literal };
                }
                else
                {
                    addresses = await Dns.GetHostAddressesAsync(host);
                }
            }
            catch (Exception ex)
            {
                return (false, $"DNS resolution failed: {ex.Message}");
            }

            if (addresses.Length == 0)
            {
                return (false, "DNS resolution returned no addresses.");
            }

            foreach (var address in addresses)
            {
                ct.ThrowIfCancellationRequested();

                if (IsBlockedAddress(address))
                {
                    _logger.LogWarning("[MCP/News] Blocked target address for {Host}: {Address}", host, address);
                    return (false, $"Resolved blocked address: {address}");
                }
            }

            _logger.LogDebug("[MCP/News] URL validated for scraping: {Host}", host);
            return (true, "");
        }

        private static bool IsLocalhostHost(string host)
        {
            return host.Equals("localhost", StringComparison.OrdinalIgnoreCase)
                || host.EndsWith(".localhost", StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsBlockedAddress(IPAddress address)
        {
            if (IPAddress.IsLoopback(address))
                return true;

            if (address.Equals(IPAddress.Any) || address.Equals(IPAddress.IPv6Any) || address.Equals(IPAddress.None))
                return true;

            if (address.AddressFamily == AddressFamily.InterNetwork)
            {
                var bytes = address.GetAddressBytes();
                return bytes[0] == 10
                    || bytes[0] == 127
                    || bytes[0] == 0
                    || (bytes[0] == 172 && bytes[1] >= 16 && bytes[1] <= 31)
                    || (bytes[0] == 192 && bytes[1] == 168)
                    || (bytes[0] == 169 && bytes[1] == 254);
            }

            if (address.AddressFamily == AddressFamily.InterNetworkV6)
            {
                if (address.Equals(IPAddress.IPv6Loopback)
                    || address.Equals(IPAddress.IPv6Any)
                    || address.Equals(IPAddress.IPv6None)
                    || address.IsIPv6LinkLocal
                    || address.IsIPv6SiteLocal
                    || address.IsIPv6Multicast)
                {
                    return true;
                }

                var bytes = address.GetAddressBytes();
                // fc00::/7 (unique local addresses)
                if ((bytes[0] & 0xFE) == 0xFC)
                    return true;
            }

            return false;
        }

        private static string BuildErrorJson(string message)
        {
            string escaped = message.Replace("\\", "\\\\").Replace("\"", "\\\"");
            return $"{{\"ok\":false,\"error\":\"{escaped}\"}}";
        }
    }
}
