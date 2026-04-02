// CONTRACT / INVARIANTS
// - The ONLY class allowed to spawn Python processes in the entire application.
// - Encapsulates a global SemaphoreSlim(3) — max 3 concurrent Python processes.
// - Invokes a domain router script (Axiom or Aether) as the single Python entrypoint per request.
// - Uses ProcessRunner with ArgumentList (injection-safe, no string concatenation).
// - Resolves python exe path via PythonPathResolver (venv, never bare "python").
// - Resolves router script paths relative to ContentRootPath.
// - Returns ProcessResult; callers handle JSON parsing and domain-specific logic.
// - Thread-safe: SemaphoreSlim gates concurrent access; no shared mutable state.
// - Never prints secrets to stdout/stderr/logs.
// - Enforces domain/action allowlist — rejects unknown routes before spawning a process.
// - Enforces per-argument and total payload size limits — prevents oversized IPC.
// - Sliding-window rate limiter prevents runaway invocation floods.

using System.Text;

namespace Aleph
{
    /// <summary>
    /// Single gateway for all Python process invocations.
    /// Routes every call through a domain router script with domain/action args.
    /// Enforces allowlists, payload size caps, and rate limits.
    /// </summary>
    public sealed class PythonDispatcherService
    {
        private readonly SemaphoreSlim _pythonGate = new(3, 3);
        private readonly string _pythonExePath;
        private readonly string _axiomRouterScriptPath;
        private readonly string _aetherRouterScriptPath;
        private readonly bool _isAvailable;
        private readonly ILogger<PythonDispatcherService> _logger;

        /// <summary>Max total bytes across all arguments for a single invocation (15 MB).</summary>
        private const int MaxTotalArgBytes = 15 * 1024 * 1024;

        /// <summary>Max bytes for any single argument (10 MB — covers metabolic JSON payloads).</summary>
        private const int MaxSingleArgBytes = 10 * 1024 * 1024;

        // ── Domain/Action Allowlist ──────────────────────────────────────
        // If Arbiter ever gains the ability to hot-swap run params, only these
        // routes can be invoked. Anything not listed here is rejected pre-spawn.
        private static readonly Dictionary<string, HashSet<string>> AllowedRoutes = new(StringComparer.OrdinalIgnoreCase)
        {
            ["market"] = new(StringComparer.OrdinalIgnoreCase) { "ingest", "parquet-read" },
            ["perception"] = new(StringComparer.OrdinalIgnoreCase) { "ingest", "snapshot" },
            ["aether"] = new(StringComparer.OrdinalIgnoreCase)
            {
                // Aether domain uses a two-hop routing: domain → sub-domain → action.
                // The first additionalArg is the sub-action, validated separately.
                "math", "ml", "sim", "macro"
            },
        };

        // ── Per-domain rate limiter tiers ────────────────────────────────
        // Different domains have fundamentally different invocation patterns:
        //   - "aether": ML/Sim/Math — high-frequency during backtest, batch
        //     ingestion, and sleep cycles. Needs a generous ceiling.
        //   - "market": Batch ingestion can fire many parquet-reads. Moderate.
        //   - "perception": Hits external APIs — keep conservative to avoid
        //     upstream throttling and billing surprises.
        //   - default: Anything else gets the strictest tier.
        private static readonly Dictionary<string, (int MaxPerWindow, TimeSpan Window)> RateTiers =
            new(StringComparer.OrdinalIgnoreCase)
        {
            ["aether"]     = (MaxPerWindow: 600, Window: TimeSpan.FromMinutes(1)),
            ["market"]     = (MaxPerWindow: 200, Window: TimeSpan.FromMinutes(1)),
            ["perception"] = (MaxPerWindow:  30, Window: TimeSpan.FromMinutes(1)),
        };
        private static readonly (int MaxPerWindow, TimeSpan Window) DefaultTier = (60, TimeSpan.FromMinutes(1));

        // ── Rate limiter state: one sliding window per domain (thread-safe via lock) ──
        private readonly object _rateLock = new();
        private readonly Dictionary<string, Queue<DateTimeOffset>> _rateWindows = new(StringComparer.OrdinalIgnoreCase);

        public bool IsAvailable => _isAvailable;

        public PythonDispatcherService(
            PythonPathResolver pythonPath,
            IHostEnvironment env,
            ILogger<PythonDispatcherService> logger)
        {
            _pythonExePath = pythonPath.ExePath;
            _isAvailable = pythonPath.IsAvailable;
            _logger = logger;

            _axiomRouterScriptPath = Path.GetFullPath(
                Path.Combine(env.ContentRootPath, "Axiom", "Python", "python_router.py"));
            _aetherRouterScriptPath = Path.GetFullPath(
                Path.Combine(env.ContentRootPath, "Aether", "Python", "aether_router.py"));

            if (_isAvailable && !File.Exists(_axiomRouterScriptPath))
            {
                _logger.LogError(
                    "[Dispatcher] Axiom Python router not found at '{RouterPath}'. " +
                    "Python dispatch will fail.", _axiomRouterScriptPath);
            }

            if (_isAvailable && !File.Exists(_aetherRouterScriptPath))
            {
                _logger.LogError(
                    "[Dispatcher] Aether Python router not found at '{RouterPath}'. " +
                    "Aether Python dispatch will fail.", _aetherRouterScriptPath);
            }
        }

        /// <summary>
        /// Run a Python command through the router.
        /// Args: python_router.py {domain} {action} {additionalArgs...}
        /// Gated by SemaphoreSlim(3). Kills on timeout.
        /// Pre-validated: domain/action allowlist, payload size, rate limit.
        /// </summary>
        public async Task<ProcessResult> RunAsync(
            string domain,
            string action,
            IReadOnlyList<string> additionalArgs,
            int timeoutMs,
            CancellationToken ct = default)
        {
            if (!_isAvailable)
            {
                return new ProcessResult(
                    false, "",
                    "Python not available. Run setup_venv.ps1 to create the venv.",
                    -1, false);
            }

            // ── Gate 1: Domain/Action Allowlist ──
            if (!IsRouteAllowed(domain, action))
            {
                _logger.LogWarning(
                    "[Dispatcher] Rejected unknown route: {Domain}/{Action}.", domain, action);
                return new ProcessResult(
                    false, "",
                    $"Route '{domain}/{action}' is not in the dispatcher allowlist.",
                    -1, false);
            }

            var safeArgs = additionalArgs ?? Array.Empty<string>();

            // ── Gate 2: Payload Size Cap ──
            var sizeCheck = ValidatePayloadSize(safeArgs);
            if (sizeCheck is not null)
            {
                _logger.LogWarning(
                    "[Dispatcher] Payload size rejected for {Domain}/{Action}: {Reason}",
                    domain, action, sizeCheck);
                return new ProcessResult(false, "", sizeCheck, -1, false);
            }

            // ── Gate 3: Per-Domain Rate Limiter ──
            var (rateMax, rateWindow) = GetRateTier(domain);
            if (!TryConsumeRateToken(domain, rateMax, rateWindow))
            {
                _logger.LogWarning(
                    "[Dispatcher] Rate limit exceeded for domain '{Domain}' ({Max}/{Window}). Rejecting {Action}.",
                    domain, rateMax, rateWindow, action);
                return new ProcessResult(
                    false, "",
                    $"Python dispatch rate limit exceeded for '{domain}' ({rateMax}/{rateWindow.TotalSeconds}s). Try again shortly.",
                    -1, false);
            }

            await _pythonGate.WaitAsync(ct);
            try
            {
                var args = BuildArguments(domain, action, safeArgs);

                _logger.LogDebug("[Dispatcher] Running: {Domain} {Action} ({ArgCount} extra args)",
                    domain, action, safeArgs.Count);

                return await ProcessRunner.RunAsync(_pythonExePath, args, timeoutMs, ct);
            }
            finally
            {
                _pythonGate.Release();
            }
        }

        // ── Allowlist check ──

        private static bool IsRouteAllowed(string domain, string action)
        {
            if (string.IsNullOrWhiteSpace(domain) || string.IsNullOrWhiteSpace(action))
                return false;

            return AllowedRoutes.TryGetValue(domain, out var actions) && actions.Contains(action);
        }

        // ── Payload size validation ──

        private static string? ValidatePayloadSize(IReadOnlyList<string> args)
        {
            long totalBytes = 0;

            for (int i = 0; i < args.Count; i++)
            {
                var arg = args[i];
                if (arg is null) continue;

                int argBytes = Encoding.UTF8.GetByteCount(arg);

                if (argBytes > MaxSingleArgBytes)
                    return $"Argument at index {i} exceeds single-arg limit ({argBytes:N0} > {MaxSingleArgBytes:N0} bytes).";

                totalBytes += argBytes;
            }

            if (totalBytes > MaxTotalArgBytes)
                return $"Total argument payload exceeds limit ({totalBytes:N0} > {MaxTotalArgBytes:N0} bytes).";

            return null;
        }

        // ── Per-domain sliding-window rate limiter ──

        private static (int MaxPerWindow, TimeSpan Window) GetRateTier(string domain)
        {
            return RateTiers.TryGetValue(domain, out var tier) ? tier : DefaultTier;
        }

        private bool TryConsumeRateToken(string domain, int maxPerWindow, TimeSpan window)
        {
            var now = DateTimeOffset.UtcNow;
            var cutoff = now - window;

            lock (_rateLock)
            {
                if (!_rateWindows.TryGetValue(domain, out var queue))
                {
                    queue = new Queue<DateTimeOffset>();
                    _rateWindows[domain] = queue;
                }

                // Evict expired entries
                while (queue.Count > 0 && queue.Peek() < cutoff)
                    queue.Dequeue();

                if (queue.Count >= maxPerWindow)
                    return false;

                queue.Enqueue(now);
                return true;
            }
        }

        /// <summary>
        /// Convenience: run market ingestion for a batch of symbols.
        /// Returns the raw ProcessResult — caller parses IngestionReport from Stdout.
        /// </summary>
        public Task<ProcessResult> RunMarketIngestAsync(
            string symbolsCsv,
            string interval,
            int lookbackDays,
            string outRoot,
            int timeoutMs,
            CancellationToken ct = default)
        {
            var extraArgs = new List<string>
            {
                "--symbols", symbolsCsv,
                "--interval", interval,
                "--lookbackDays", lookbackDays.ToString(),
                "--outRoot", outRoot
            };

            return RunAsync("market", "ingest", extraArgs, timeoutMs, ct);
        }

        /// <summary>
        /// Convenience: run perception ingest (macro proxies, calendar, headlines).
        /// Returns the raw ProcessResult — caller parses JSON from Stdout.
        /// </summary>
        public Task<ProcessResult> RunPerceptionIngestAsync(
            int lookbackDays,
            int headlineLimit,
            int calendarHorizonDays,
            int timeoutMs,
            CancellationToken ct = default)
        {
            var extraArgs = new List<string>
            {
                "--lookbackDays", lookbackDays.ToString(),
                "--headlineLimit", headlineLimit.ToString(),
                "--calendarHorizonDays", calendarHorizonDays.ToString()
            };

            return RunAsync("perception", "ingest", extraArgs, timeoutMs, ct);
        }

        /// <summary>
        /// Convenience: read local perception snapshot (no network calls).
        /// Returns the raw ProcessResult — caller parses JSON from Stdout.
        /// </summary>
        public Task<ProcessResult> RunPerceptionSnapshotAsync(
            int headlineLimit,
            int timeoutMs,
            CancellationToken ct = default)
        {
            var extraArgs = new List<string>
            {
                "--headlineLimit", headlineLimit.ToString()
            };

            return RunAsync("perception", "snapshot", extraArgs, timeoutMs, ct);
        }

        /// <summary>
        /// Convenience: read Parquet data for a symbol from the local data lake.
        /// Returns the raw ProcessResult — caller parses JSON from Stdout.
        /// </summary>
        public Task<ProcessResult> RunParquetReadAsync(
            string symbol,
            int days,
            string dataRoot,
            int timeoutMs,
            CancellationToken ct = default)
        {
            var extraArgs = new List<string>
            {
                "--symbol", symbol,
                "--days", days.ToString(),
                "--dataRoot", dataRoot
            };

            return RunAsync("market", "parquet-read", extraArgs, timeoutMs, ct);
        }

        private List<string> BuildArguments(
            string domain,
            string action,
            IReadOnlyList<string> additionalArgs)
        {
            if (domain.Equals("aether", StringComparison.OrdinalIgnoreCase))
            {
                if (additionalArgs.Count == 0)
                {
                    return new List<string>
                    {
                        _aetherRouterScriptPath,
                        action,
                        "__missing_action__"
                    };
                }

                var rewritten = new List<string>
                {
                    _aetherRouterScriptPath,
                    action,
                    additionalArgs[0]
                };

                for (var i = 1; i < additionalArgs.Count; i++)
                {
                    rewritten.Add(additionalArgs[i]);
                }

                return rewritten;
            }

            var args = new List<string> { _axiomRouterScriptPath, domain, action };
            args.AddRange(additionalArgs);
            return args;
        }
    }
}
