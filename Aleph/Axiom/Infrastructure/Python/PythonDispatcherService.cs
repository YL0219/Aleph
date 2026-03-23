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



namespace Aleph
{
    /// <summary>
    /// Single gateway for all Python process invocations.
    /// Routes every call through a domain router script with domain/action args.
    /// </summary>
    public sealed class PythonDispatcherService
    {
        private readonly SemaphoreSlim _pythonGate = new(3, 3);
        private readonly string _pythonExePath;
        private readonly string _axiomRouterScriptPath;
        private readonly string _aetherRouterScriptPath;
        private readonly bool _isAvailable;
        private readonly ILogger<PythonDispatcherService> _logger;

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

            var safeArgs = additionalArgs ?? Array.Empty<string>();

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
