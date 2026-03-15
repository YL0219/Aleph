using System.Text.Json;

namespace Aleph;

/// <summary>
/// Persists metabolized output to the data lake for future ML / trading reuse.
///
/// Artifact layout:
///   data_lake/metabolism/symbol={SYM}/interval={INTERVAL}/latest.json
///
/// The JSON artifact contains the full canonical math output enriched with
/// metabolic version and provenance metadata. Downstream consumers can load
/// this file instead of re-running the indicator pipeline.
/// </summary>
public sealed class MetabolicArtifactWriter
{
    private readonly IHostEnvironment _env;
    private readonly ILogger<MetabolicArtifactWriter> _logger;

    public MetabolicArtifactWriter(
        IHostEnvironment env,
        ILogger<MetabolicArtifactWriter> logger)
    {
        _env = env;
        _logger = logger;
    }

    /// <summary>
    /// Write the metabolized artifact to the data lake.
    /// Returns the relative path to the written artifact, or null on failure.
    /// </summary>
    public async Task<string?> WriteAsync(
        string symbol,
        string interval,
        string canonicalMathJson,
        string metabolicVersion,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(symbol) || string.IsNullOrWhiteSpace(canonicalMathJson))
            return null;

        var normalizedSymbol = symbol.Trim().ToUpperInvariant();
        var normalizedInterval = string.IsNullOrWhiteSpace(interval) ? "1d" : interval.Trim().ToLowerInvariant();

        // Build output path consistent with data lake conventions
        var relativePath = Path.Combine(
            "data_lake", "metabolism",
            $"symbol={normalizedSymbol}",
            $"interval={normalizedInterval}",
            "latest.json");

        var absolutePath = Path.Combine(_env.ContentRootPath, relativePath);

        try
        {
            // Ensure directory exists
            var dir = Path.GetDirectoryName(absolutePath);
            if (!string.IsNullOrWhiteSpace(dir))
                Directory.CreateDirectory(dir);

            // Enrich with metabolic envelope metadata
            var envelope = BuildEnvelope(canonicalMathJson, normalizedSymbol, normalizedInterval, metabolicVersion);

            await File.WriteAllTextAsync(absolutePath, envelope, ct);

            _logger.LogDebug(
                "[Liver/Artifact] Wrote metabolized artifact for {Symbol}/{Interval} → {Path}",
                normalizedSymbol, normalizedInterval, relativePath);

            return relativePath;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "[Liver/Artifact] Failed to write artifact for {Symbol}/{Interval} at {Path}.",
                normalizedSymbol, normalizedInterval, absolutePath);
            return null;
        }
    }

    /// <summary>
    /// Wraps the canonical math JSON with metabolic provenance metadata.
    /// </summary>
    private static string BuildEnvelope(
        string canonicalJson,
        string symbol,
        string interval,
        string metabolicVersion)
    {
        using var doc = JsonDocument.Parse(canonicalJson);

        var options = new JsonWriterOptions { Indented = true };

        using var ms = new MemoryStream();
        using (var writer = new Utf8JsonWriter(ms, options))
        {
            writer.WriteStartObject();

            // Metabolic envelope header
            writer.WriteString("metabolic_version", metabolicVersion);
            writer.WriteString("symbol", symbol);
            writer.WriteString("interval", interval);
            writer.WriteString("digested_at_utc", DateTimeOffset.UtcNow.ToString("o"));
            writer.WriteString("organ", "Liver");

            // Embed the full canonical math payload
            writer.WritePropertyName("canonical_analysis");
            doc.RootElement.WriteTo(writer);

            writer.WriteEndObject();
        }

        return System.Text.Encoding.UTF8.GetString(ms.ToArray());
    }
}
