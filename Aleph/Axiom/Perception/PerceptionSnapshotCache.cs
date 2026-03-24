// CONTRACT / INVARIANTS
// - Singleton service providing cached perception context for the Liver.
// - Reads directly from the local data lake JSON files — NO Python subprocess needed.
// - Thread-safe: lock-based double-check cache refresh.
// - Graceful degradation: missing/corrupt/stale files produce honest status, never crashes.
// - Adding new perception sections requires ONLY adding a reader here — no downstream changes.
//
// Data lake files read:
//   data_lake/perception/manifest.json           (overall metadata)
//   data_lake/macro/proxies/summary.json          (proxy latest closes)
//   data_lake/macro/calendar/latest.json          (economic calendar events)
//   data_lake/macro/headlines/latest.json         (macro news headlines)

using System.Text.Json;
using System.Text.Json.Serialization;

namespace Aleph;

/// <summary>
/// Cached perception snapshot accessor. Reads local data lake files and builds
/// a <see cref="MetabolicMacroContext"/> for the Liver to attach to metabolic events.
///
/// One instance per application lifetime. Results cached with a configurable TTL
/// so that all symbols in an ingestion batch share the same perception context.
/// </summary>
public sealed class PerceptionSnapshotCache
{
    // Staleness thresholds — must match perception_snapshot.py conventions
    private const double ProxyStaleHours = 26;
    private const double CalendarStaleHours = 72;
    private const double HeadlineStaleHours = 6;
    private static readonly JsonSerializerOptions PerceptionJsonOptions = new()
    {
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
    };

    private readonly string _manifestPath;
    private readonly string _proxySummaryPath;
    private readonly string _calendarPath;
    private readonly string _headlinesPath;
    private readonly TimeSpan _cacheTtl;
    private readonly ILogger<PerceptionSnapshotCache> _logger;

    private MetabolicMacroContext? _cached;
    private DateTimeOffset _cachedAt = DateTimeOffset.MinValue;
    private readonly object _lock = new();

    public PerceptionSnapshotCache(
        IHostEnvironment env,
        IConfiguration config,
        ILogger<PerceptionSnapshotCache> logger)
    {
        var root = env.ContentRootPath;
        _manifestPath = Path.GetFullPath(Path.Combine(root, "data_lake", "perception", "manifest.json"));
        _proxySummaryPath = Path.GetFullPath(Path.Combine(root, "data_lake", "macro", "proxies", "summary.json"));
        _calendarPath = Path.GetFullPath(Path.Combine(root, "data_lake", "macro", "calendar", "latest.json"));
        _headlinesPath = Path.GetFullPath(Path.Combine(root, "data_lake", "macro", "headlines", "latest.json"));

        var ttlSeconds = int.TryParse(config["Axiom:Perception:CacheTtlSeconds"], out var t) ? t : 120;
        _cacheTtl = TimeSpan.FromSeconds(ttlSeconds);
        _logger = logger;
    }

    /// <summary>
    /// Get the current perception context. Returns a cached snapshot if fresh,
    /// otherwise refreshes from disk. Never throws — returns degraded context on error.
    /// </summary>
    public MetabolicMacroContext GetSnapshot()
    {
        lock (_lock)
        {
            if (_cached is not null && DateTimeOffset.UtcNow - _cachedAt < _cacheTtl)
                return _cached;
        }

        var fresh = BuildFromDisk();

        lock (_lock)
        {
            _cached = fresh;
            _cachedAt = DateTimeOffset.UtcNow;
        }

        return fresh;
    }

    /// <summary>
    /// Force cache invalidation. Called when a new perception refresh completes.
    /// </summary>
    public void Invalidate()
    {
        lock (_lock)
        {
            _cached = null;
            _cachedAt = DateTimeOffset.MinValue;
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Disk readers — one per section
    // ═══════════════════════════════════════════════════════════════════

    private MetabolicMacroContext BuildFromDisk()
    {
        var sections = new Dictionary<string, MacroContextSection>();

        // ── Manifest ──
        bool manifestPresent = File.Exists(_manifestPath);
        string? manifestFetchedAt = null;
        if (manifestPresent)
        {
            manifestFetchedAt = ExtractTimestamp(_manifestPath, "fetchedAtUtc");
        }

        // ── Proxies ──
        sections["proxies"] = ReadSection(
            "proxies", _proxySummaryPath, ProxyStaleHours, "fetchedAtUtc");

        // ── Calendar ──
        sections["calendar"] = ReadSection(
            "calendar", _calendarPath, CalendarStaleHours, "fetchedAtUtc");

        // ── Headlines ──
        sections["headlines"] = ReadSection(
            "headlines", _headlinesPath, HeadlineStaleHours, "fetchedAtUtc");

        // ── Compute envelope ──
        int available = sections.Values.Count(s => s.Status is "fresh" or "stale");
        bool anyStale = sections.Values.Any(s => s.Status == "stale");

        string freshness = available == 0 ? "missing"
            : anyStale ? "partial"
            : "fresh";

        var result = new MetabolicMacroContext
        {
            SnapshotAtUtc = DateTimeOffset.UtcNow.ToString("o"),
            Freshness = freshness,
            ManifestPresent = manifestPresent,
            AnyStale = anyStale,
            SectionsAvailable = available,
            Sections = sections,
        };

        _logger.LogDebug(
            "[PerceptionCache] Snapshot built: freshness={Freshness}, sections={Available}/{Total}, stale={Stale}",
            freshness, available, sections.Count, anyStale);

        return result;
    }

    private MacroContextSection ReadSection(
        string name,
        string filePath,
        double staleHours,
        string timestampKey)
    {
        if (!File.Exists(filePath))
        {
            return new MacroContextSection
            {
                Name = name,
                Status = "missing",
            };
        }

        string rawJson;
        try
        {
            rawJson = File.ReadAllText(filePath);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "[PerceptionCache] Failed to read {Section} from {Path}", name, filePath);
            return new MacroContextSection
            {
                Name = name,
                Status = "error",
                Warnings = new[] { $"Read failed: {ex.Message}" },
            };
        }

        // Extract timestamp for staleness check
        string? fetchedAt = ExtractTimestampFromJson(rawJson, timestampKey);
        string? provider = ExtractStringFromJson(rawJson, "provider");

        string status = DetermineFreshness(fetchedAt, staleHours);

        return new MacroContextSection
        {
            Name = name,
            Status = status,
            FetchedAtUtc = fetchedAt,
            Provider = provider,
            PayloadJson = rawJson,
        };
    }

    // ═══════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════

    private static string DetermineFreshness(string? fetchedAtUtc, double staleHours)
    {
        if (string.IsNullOrEmpty(fetchedAtUtc))
            return "stale"; // can't determine age → conservative

        if (!DateTimeOffset.TryParse(fetchedAtUtc, out var fetched))
            return "stale";

        var age = DateTimeOffset.UtcNow - fetched;
        return age.TotalHours > staleHours ? "stale" : "fresh";
    }

    private string? ExtractTimestamp(string filePath, string key)
    {
        try
        {
            var json = File.ReadAllText(filePath);
            return ExtractTimestampFromJson(json, key);
        }
        catch
        {
            return null;
        }
    }

    private static string? ExtractTimestampFromJson(string json, string key)
    {
        try
        {
            var root = JsonSerializer.Deserialize<JsonElement>(json, PerceptionJsonOptions);
            return root.ValueKind == JsonValueKind.Object
                && root.TryGetProperty(key, out var prop)
                && prop.ValueKind == JsonValueKind.String
                ? prop.GetString()
                : null;
        }
        catch
        {
            return null;
        }
    }

    private static string? ExtractStringFromJson(string json, string key)
    {
        try
        {
            var root = JsonSerializer.Deserialize<JsonElement>(json, PerceptionJsonOptions);
            return root.ValueKind == JsonValueKind.Object
                && root.TryGetProperty(key, out var prop)
                && prop.ValueKind == JsonValueKind.String
                ? prop.GetString()
                : null;
        }
        catch
        {
            return null;
        }
    }
}
