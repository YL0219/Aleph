// CONTRACT / INVARIANTS
// - Runs external processes using ArgumentList (no shell interpretation, injection-safe).
// - Drains stdout + stderr concurrently to prevent OS pipe-buffer deadlocks.
// - Supports timeout via CancelAfter; kills entire process tree on timeout.
// - Returns structured ProcessResult; callers decide error handling.
// - Thread-safe: no shared mutable state. Safe to call from parallel tasks.
// - Enforces max stdout/stderr size to prevent memory-bomb attacks from rogue scripts.
// - Validates executable path is rooted and exists before spawning.

using System.Diagnostics;
using System.Text;

namespace Aleph
{
    /// <summary>
    /// Structured result from a process execution.
    /// </summary>
    public sealed record ProcessResult(
        bool Success,
        string Stdout,
        string Stderr,
        int ExitCode,
        bool TimedOut,
        bool StdoutTruncated = false);

    /// <summary>
    /// Centralized helper for running external processes (Python, curl, etc.).
    /// All arguments go through ProcessStartInfo.ArgumentList to avoid
    /// shell-interpretation and injection/quoting issues.
    /// </summary>
    public static class ProcessRunner
    {
        /// <summary>Max bytes for stdout before truncation (10 MB). Prevents memory bombs.</summary>
        private const int MaxStdoutBytes = 10 * 1024 * 1024;

        /// <summary>Max bytes for stderr before truncation (1 MB).</summary>
        private const int MaxStderrBytes = 1 * 1024 * 1024;

        /// <summary>
        /// Spawns a process, drains stdout/stderr, enforces a timeout, and returns results.
        /// </summary>
        /// <param name="fileName">Executable path (e.g., python.exe, curl.exe). Must be rooted.</param>
        /// <param name="arguments">Argument list — each element is one logical argument.</param>
        /// <param name="timeoutMs">Hard timeout in milliseconds. Process tree is killed on expiry.</param>
        /// <param name="ct">External cancellation token (e.g., app shutdown).</param>
        public static async Task<ProcessResult> RunAsync(
            string fileName,
            IReadOnlyList<string> arguments,
            int timeoutMs,
            CancellationToken ct = default)
        {
            // ── Gate 1: Validate executable path ──
            if (string.IsNullOrWhiteSpace(fileName))
                return new ProcessResult(false, "", "Executable path is empty.", -1, false);

            if (!Path.IsPathRooted(fileName))
                return new ProcessResult(false, "", "Executable path must be absolute (rooted).", -1, false);

            if (!File.Exists(fileName))
                return new ProcessResult(false, "", $"Executable not found: {fileName}", -1, false);

            var psi = new ProcessStartInfo
            {
                FileName = fileName,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            foreach (var arg in arguments)
                psi.ArgumentList.Add(arg);

            Process? process;
            try
            {
                process = Process.Start(psi);
            }
            catch (Exception ex)
            {
                return new ProcessResult(false, "", $"Failed to start process: {ex.Message}", -1, false);
            }

            if (process == null)
                return new ProcessResult(false, "", "Process.Start returned null.", -1, false);

            using (process)
            {
                // ── Drain both streams concurrently with size caps ──
                // Uses bounded readers to prevent a rogue script from blowing up memory.
                var stdoutTask = ReadBoundedAsync(process.StandardOutput, MaxStdoutBytes);
                var stderrTask = ReadBoundedAsync(process.StandardError, MaxStderrBytes);

                using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                cts.CancelAfter(timeoutMs);

                try
                {
                    await process.WaitForExitAsync(cts.Token);
                }
                catch (OperationCanceledException)
                {
                    try { process.Kill(entireProcessTree: true); } catch { }

                    bool appShutdown = ct.IsCancellationRequested;
                    return new ProcessResult(
                        false, "",
                        appShutdown ? "Operation cancelled." : $"Process timed out after {timeoutMs / 1000}s.",
                        -1,
                        TimedOut: !appShutdown);
                }

                var (stdout, stdoutTruncated) = await stdoutTask;
                var (stderr, _) = await stderrTask;

                return new ProcessResult(
                    Success: process.ExitCode == 0 && !stdoutTruncated,
                    Stdout: stdout.TrimEnd(),
                    Stderr: stdoutTruncated
                        ? $"[TRUNCATED] stdout exceeded {MaxStdoutBytes / (1024 * 1024)}MB cap. Output is malformed. " + stderr.TrimEnd()
                        : stderr.TrimEnd(),
                    ExitCode: process.ExitCode,
                    TimedOut: false,
                    StdoutTruncated: stdoutTruncated);
            }
        }

        /// <summary>
        /// Reads from a StreamReader up to maxBytes, then discards the rest.
        /// Returns (content, wasTruncated).
        /// </summary>
        private static async Task<(string Content, bool Truncated)> ReadBoundedAsync(
            StreamReader reader, int maxBytes)
        {
            var sb = new StringBuilder();
            var buffer = new char[8192];
            int totalBytes = 0;
            bool truncated = false;

            while (true)
            {
                int charsRead = await reader.ReadAsync(buffer, 0, buffer.Length);
                if (charsRead == 0) break;

                int byteCount = Encoding.UTF8.GetByteCount(buffer, 0, charsRead);

                if (totalBytes + byteCount > maxBytes)
                {
                    // Take only what fits
                    int remaining = maxBytes - totalBytes;
                    if (remaining > 0)
                    {
                        // Approximate char count for remaining bytes
                        int approxChars = Math.Min(charsRead, remaining);
                        sb.Append(buffer, 0, approxChars);
                    }
                    truncated = true;
                    // Keep draining to prevent pipe deadlock, but discard
                    while (await reader.ReadAsync(buffer, 0, buffer.Length) > 0) { }
                    break;
                }

                sb.Append(buffer, 0, charsRead);
                totalBytes += byteCount;
            }

            return (sb.ToString(), truncated);
        }
    }
}
