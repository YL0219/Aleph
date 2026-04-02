using System.Text.Json;
using System.Threading.Channels;

namespace Aleph;

/// <summary>
/// The Liver — metabolic processing organ of the Aleph organism.
///
/// Subscribes to the AlephBus for MarketDataEvent blood cells, digests the
/// persisted canonical market data through Aether's math pipeline, and publishes
/// refined MetabolicEvent blood cells back into the bloodstream.
///
/// Key design principles:
///   - Reuses Aether's canonical math/indicator pipeline (no competing engine)
///   - Processes persisted data only (parquet artifacts), never fetches live data
///   - Publishes + persists metabolized output for downstream ML/Trading organs
///   - Market-focused in this sprint, structured for future data families
/// </summary>
public sealed class LiverService : BackgroundService, IAlephOrgan
{
    private const string MetabolicVersionV1 = "1.0.0";
    private const int MaxConcurrentDigestions = 2;
    private const int DigestionTimeoutMs = 60_000;

    private readonly IAlephBus _bus;
    private readonly IAether _aether;
    private readonly PerceptionSnapshotCache _perceptionCache;
    private readonly MetabolicArtifactWriter _artifactWriter;
    private readonly Quarantine _quarantine;
    private readonly ILogger<LiverService> _logger;

    private volatile bool _isActive;

    public string OrganName => "Liver";

    public IReadOnlyList<string> EventInterests { get; } =
        new[] { nameof(MarketDataEvent) }.AsReadOnly();

    public bool IsActive => _isActive;

    public LiverService(
        IAlephBus bus,
        IAether aether,
        PerceptionSnapshotCache perceptionCache,
        MetabolicArtifactWriter artifactWriter,
        Quarantine quarantine,
        ILogger<LiverService> logger)
    {
        _bus = bus;
        _aether = aether;
        _perceptionCache = perceptionCache;
        _artifactWriter = artifactWriter;
        _quarantine = quarantine;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogDebug("[Liver] Metabolic organ starting. Subscribing to bloodstream...");

        var reader = _bus.Subscribe(
            OrganName,
            static evt => evt is MarketDataEvent);

        _isActive = true;

        _logger.LogDebug("[Liver] Active. Awaiting MarketDataEvent blood cells.");

        try
        {
            // Use a semaphore to limit concurrent digestions
            using var throttle = new SemaphoreSlim(MaxConcurrentDigestions);

            await foreach (var evt in reader.ReadAllAsync(stoppingToken))
            {
                if (evt is not MarketDataEvent mde)
                    continue;

                // Fire-and-forget with throttle — don't block the subscription loop
                await throttle.WaitAsync(stoppingToken);

                _ = Task.Run(async () =>
                {
                    try
                    {
                        await DigestMarketDataAsync(mde, stoppingToken);
                    }
                    catch (OperationCanceledException) { /* shutting down */ }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex,
                            "[Liver] Unhandled error digesting {Symbol}/{Interval}.",
                            mde.Symbol, mde.Interval);
                    }
                    finally
                    {
                        throttle.Release();
                    }
                }, stoppingToken);
            }
        }
        catch (OperationCanceledException) { /* expected on shutdown */ }
        finally
        {
            _isActive = false;
            _logger.LogInformation("[Liver] Metabolic organ stopped.");
        }
    }

    /// <summary>
    /// Digest a single MarketDataEvent: screen → run canonical math → build MetabolicEvent → publish + persist.
    /// </summary>
    private async Task DigestMarketDataAsync(MarketDataEvent mde, CancellationToken ct)
    {
        // ── Step 1: BloodFilter immune screening ──
        var screening = BloodFilter.Screen(mde);
        if (!screening.IsHealthy)
        {
            _quarantine.Isolate(OrganName, mde, screening);
            return;
        }

        _logger.LogDebug(
            "[Liver] Digesting {Symbol}/{Interval} (source event {EventId}).",
            mde.Symbol, mde.Interval, mde.EventId);

        // ── Step 2: Run Aether's canonical math pipeline ──
        //
        // This is the critical reuse point. We call IAether.Math.EvaluateIndicatorsAsync
        // which routes through aether_router.py → math_manager.py → quant/analysis.py.
        // The Liver does NOT maintain a second indicator engine.
        //
        var mathRequest = new MathIndicatorsRequest(
            Symbol: mde.Symbol,
            Days: 0, // 0 = use all available data
            Timeframe: string.IsNullOrWhiteSpace(mde.Interval) ? "1d" : mde.Interval);

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(DigestionTimeoutMs);

        AetherJsonResult mathResult;
        try
        {
            mathResult = await _aether.Math.EvaluateIndicatorsAsync(mathRequest, timeoutCts.Token);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            _logger.LogWarning(
                "[Liver] Digestion timed out for {Symbol}/{Interval} after {Timeout}ms.",
                mde.Symbol, mde.Interval, DigestionTimeoutMs);
            return;
        }

        if (!mathResult.Success || string.IsNullOrWhiteSpace(mathResult.PayloadJson))
        {
            _logger.LogWarning(
                "[Liver] Canonical math failed for {Symbol}/{Interval}: {Error}",
                mde.Symbol, mde.Interval, mathResult.Error ?? "empty payload");
            return;
        }

        // ── Step 3: Parse the canonical math output ──
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(mathResult.PayloadJson);
        }
        catch (JsonException ex)
        {
            _logger.LogWarning(ex,
                "[Liver] Failed to parse math JSON for {Symbol}/{Interval}.",
                mde.Symbol, mde.Interval);
            return;
        }

        using (doc)
        {
            var root = doc.RootElement;

            if (!root.TryGetProperty("ok", out var okProp) || !okProp.GetBoolean())
            {
                var error = root.TryGetProperty("error", out var errProp)
                    ? errProp.GetString() ?? "unknown"
                    : "unknown";
                _logger.LogWarning(
                    "[Liver] Math pipeline returned ok=false for {Symbol}/{Interval}: {Error}",
                    mde.Symbol, mde.Interval, error);
                return;
            }

            // ── Step 4: Build MetabolicEvent from canonical output ──
            //
            // The perception cache provides the current macro context (proxies,
            // calendar, headlines) read from the local data lake. The cache is
            // shared across all symbols in an ingestion batch — one disk read,
            // many events enriched.
            //
            MetabolicMacroContext? macroCtx = null;
            MetabolicContextCoverage? coverage = null;
            try
            {
                macroCtx = _perceptionCache.GetSnapshot();
                coverage = BuildCoverage(macroCtx);
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex,
                    "[Liver] Perception snapshot unavailable for {Symbol}. Proceeding without macro context.",
                    mde.Symbol);
            }

            var metabolicEvent = BuildMetabolicEvent(mde, root, macroCtx, coverage);

            // ── Step 5: Persist the metabolized artifact ──
            string? artifactPath = null;
            try
            {
                artifactPath = await _artifactWriter.WriteAsync(
                    mde.Symbol,
                    mde.Interval,
                    mathResult.PayloadJson,
                    MetabolicVersionV1,
                    ct);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex,
                    "[Liver] Failed to persist metabolized artifact for {Symbol}/{Interval}.",
                    mde.Symbol, mde.Interval);
                // Non-fatal: still publish the event
            }

            // Attach artifact path if persistence succeeded
            var finalEvent = artifactPath is not null
                ? metabolicEvent with { MetabolizedArtifactPath = artifactPath }
                : metabolicEvent;

            // ── Step 6: Publish MetabolicEvent into the bloodstream ──
            await _bus.PublishAsync(finalEvent, ct);

            _logger.LogInformation(
                "[Liver] {Symbol}/{Interval} → Bias={Bias} Conf={Confidence:F2} Rows={Rows}",
                mde.Symbol, mde.Interval, finalEvent.Bias, finalEvent.Confidence,
                finalEvent.RowCount);
        }
    }

    // ─── JSON → MetabolicEvent Mapping ───────────────────────────────

    private static MetabolicEvent BuildMetabolicEvent(
        MarketDataEvent mde,
        JsonElement root,
        MetabolicMacroContext? macroContext,
        MetabolicContextCoverage? coverage)
    {
        var asof = SafeString(root, "asof_utc") ?? DateTimeOffset.UtcNow.ToString("o");
        var dataQuality = root.TryGetProperty("data_quality", out var dq) ? dq : default;
        var snapshotEl = root.TryGetProperty("snapshot", out var sn) ? sn : default;
        var factorEl = root.TryGetProperty("factor_scores", out var fs) ? fs : default;
        var compositeEl = root.TryGetProperty("composite", out var co) ? co : default;
        var conclusionEl = root.TryGetProperty("conclusion", out var cl) ? cl : default;
        var windowsEl = root.TryGetProperty("recent_windows", out var rw) ? rw : default;

        var warnings = new List<string>();
        if (dataQuality.ValueKind == JsonValueKind.Object &&
            dataQuality.TryGetProperty("warnings", out var warnArr) &&
            warnArr.ValueKind == JsonValueKind.Array)
        {
            foreach (var w in warnArr.EnumerateArray())
            {
                var ws = w.GetString();
                if (!string.IsNullOrWhiteSpace(ws))
                    warnings.Add(ws);
            }
        }

        return new MetabolicEvent
        {
            OccurredAtUtc = DateTimeOffset.UtcNow,
            Source = "Liver",
            Kind = "metabolic_digest",
            Severity = PulseSeverity.Normal,
            Tags = new[] { "market", "digest" },
            CorrelationId = mde.CorrelationId,
            CausationId = mde.EventId,

            Symbol = mde.Symbol,
            Interval = mde.Interval,
            AsOfUtc = asof,
            SourceEventId = mde.EventId,
            SourceParquetPath = mde.ParquetPath,
            MetabolicVersion = MetabolicVersionV1,

            RowCount = SafeInt(dataQuality, "rows"),
            EnoughForLongTrend = SafeBool(dataQuality, "enough_for_long_trend"),
            Warnings = warnings.AsReadOnly(),

            Snapshot = ParseSnapshot(snapshotEl),
            FactorScores = ParseFactorScores(factorEl),
            Composite = ParseComposite(compositeEl),

            Bias = SafeString(conclusionEl, "bias") ?? "unknown",
            Confidence = SafeDouble(compositeEl, "confidence") ?? 0,
            KeyDrivers = ParseStringArray(conclusionEl, "key_drivers"),
            Risks = ParseStringArray(conclusionEl, "risks"),

            RecentWindows = ParseRecentWindows(windowsEl),
            MacroContext = macroContext,
            ContextCoverage = coverage,
        };
    }

    /// <summary>
    /// Build context coverage from the macro context sections.
    /// Reports which perception sections were available.
    /// </summary>
    private static MetabolicContextCoverage BuildCoverage(MetabolicMacroContext? macro)
    {
        if (macro is null)
        {
            return new MetabolicContextCoverage
            {
                MissingContextReasons = new[] { "perception_snapshot_unavailable" },
            };
        }

        var missing = new List<string>();

        bool hadProxies = macro.Sections.TryGetValue("proxies", out var p)
                          && p.Status is "fresh" or "stale";
        bool hadCalendar = macro.Sections.TryGetValue("calendar", out var c)
                           && c.Status is "fresh" or "stale";
        bool hadHeadlines = macro.Sections.TryGetValue("headlines", out var h)
                            && h.Status is "fresh" or "stale";

        if (!hadProxies) missing.Add("proxies_missing");
        if (!hadCalendar) missing.Add("calendar_missing");
        if (!hadHeadlines) missing.Add("headlines_missing");

        return new MetabolicContextCoverage
        {
            HadProxyContext = hadProxies,
            HadCalendarContext = hadCalendar,
            HadHeadlineContext = hadHeadlines,
            MissingContextReasons = missing.AsReadOnly(),
        };
    }

    private static MetabolicSnapshot ParseSnapshot(JsonElement el)
    {
        if (el.ValueKind != JsonValueKind.Object)
            return new MetabolicSnapshot();

        var macdEl = el.TryGetProperty("macd", out var m) ? m : default;
        var bbEl = el.TryGetProperty("bollinger", out var b) ? b : default;

        return new MetabolicSnapshot
        {
            Price = SafeDouble(el, "price"),
            Sma20 = SafeDouble(el, "sma_20"),
            Sma50 = SafeDouble(el, "sma_50"),
            Sma200 = SafeDouble(el, "sma_200"),
            Ema12 = SafeDouble(el, "ema_12"),
            Ema26 = SafeDouble(el, "ema_26"),
            Rsi14 = SafeDouble(el, "rsi_14"),
            Macd = macdEl.ValueKind == JsonValueKind.Object
                ? new MetabolicMacd
                {
                    Line = SafeDouble(macdEl, "line"),
                    Signal = SafeDouble(macdEl, "signal"),
                    Histogram = SafeDouble(macdEl, "histogram"),
                }
                : null,
            Bollinger = bbEl.ValueKind == JsonValueKind.Object
                ? new MetabolicBollinger
                {
                    Mid = SafeDouble(bbEl, "mid"),
                    Upper = SafeDouble(bbEl, "upper"),
                    Lower = SafeDouble(bbEl, "lower"),
                    Bandwidth = SafeDouble(bbEl, "bandwidth"),
                }
                : null,
            Atr14 = SafeDouble(el, "atr_14"),
            AtrPct = SafeDouble(el, "atr_pct"),
            Volatility20 = SafeDouble(el, "volatility_20"),
            VolumeSma20 = SafeDouble(el, "volume_sma_20"),
        };
    }

    private static MetabolicFactorScores ParseFactorScores(JsonElement el)
    {
        return new MetabolicFactorScores
        {
            Trend = ParseFactor(el, "trend"),
            Momentum = ParseFactor(el, "momentum"),
            Volatility = ParseFactor(el, "volatility"),
            Participation = ParseFactor(el, "participation"),
        };
    }

    private static MetabolicFactor ParseFactor(JsonElement parent, string name)
    {
        if (parent.ValueKind != JsonValueKind.Object ||
            !parent.TryGetProperty(name, out var el) ||
            el.ValueKind != JsonValueKind.Object)
        {
            return new MetabolicFactor { Score = 0, Label = "unknown" };
        }

        return new MetabolicFactor
        {
            Score = SafeDouble(el, "score") ?? 0,
            Label = SafeString(el, "label") ?? "unknown",
            Reason = SafeString(el, "reason"),
        };
    }

    private static MetabolicComposite ParseComposite(JsonElement el)
    {
        if (el.ValueKind != JsonValueKind.Object)
        {
            return new MetabolicComposite
            {
                BullishProbability = 0.33,
                BearishProbability = 0.33,
                NeutralProbability = 0.34,
                Confidence = 0,
            };
        }

        return new MetabolicComposite
        {
            BullishProbability = SafeDouble(el, "bullish_probability") ?? 0.33,
            BearishProbability = SafeDouble(el, "bearish_probability") ?? 0.33,
            NeutralProbability = SafeDouble(el, "neutral_probability") ?? 0.34,
            Confidence = SafeDouble(el, "confidence") ?? 0,
        };
    }

    private static MetabolicRecentWindows? ParseRecentWindows(JsonElement el)
    {
        if (el.ValueKind != JsonValueKind.Object)
            return null;

        return new MetabolicRecentWindows
        {
            Close = ParseDoubleArray(el, "close"),
            Rsi14 = ParseDoubleArray(el, "rsi_14"),
            MacdHistogram = ParseDoubleArray(el, "macd_histogram"),
            AtrPct = ParseDoubleArray(el, "atr_pct"),
        };
    }

    // ─── Safe JSON Helpers ───────────────────────────────────────────

    private static double? SafeDouble(JsonElement el, string prop)
    {
        if (el.ValueKind != JsonValueKind.Object)
            return null;
        if (!el.TryGetProperty(prop, out var val))
            return null;
        return val.ValueKind switch
        {
            JsonValueKind.Number => val.GetDouble(),
            JsonValueKind.Null => null,
            _ => null,
        };
    }

    private static string? SafeString(JsonElement el, string prop)
    {
        if (el.ValueKind != JsonValueKind.Object)
            return null;
        if (!el.TryGetProperty(prop, out var val))
            return null;
        return val.ValueKind == JsonValueKind.String ? val.GetString() : null;
    }

    private static int SafeInt(JsonElement el, string prop)
    {
        if (el.ValueKind != JsonValueKind.Object)
            return 0;
        if (!el.TryGetProperty(prop, out var val))
            return 0;
        return val.ValueKind == JsonValueKind.Number ? val.GetInt32() : 0;
    }

    private static bool SafeBool(JsonElement el, string prop)
    {
        if (el.ValueKind != JsonValueKind.Object)
            return false;
        if (!el.TryGetProperty(prop, out var val))
            return false;
        return val.ValueKind is JsonValueKind.True or JsonValueKind.False && val.GetBoolean();
    }

    private static IReadOnlyList<string> ParseStringArray(JsonElement el, string prop)
    {
        if (el.ValueKind != JsonValueKind.Object)
            return Array.Empty<string>();
        if (!el.TryGetProperty(prop, out var arr) || arr.ValueKind != JsonValueKind.Array)
            return Array.Empty<string>();

        var list = new List<string>();
        foreach (var item in arr.EnumerateArray())
        {
            var s = item.GetString();
            if (!string.IsNullOrWhiteSpace(s))
                list.Add(s);
        }
        return list.AsReadOnly();
    }

    private static IReadOnlyList<double?> ParseDoubleArray(JsonElement el, string prop)
    {
        if (el.ValueKind != JsonValueKind.Object)
            return Array.Empty<double?>();
        if (!el.TryGetProperty(prop, out var arr) || arr.ValueKind != JsonValueKind.Array)
            return Array.Empty<double?>();

        var list = new List<double?>();
        foreach (var item in arr.EnumerateArray())
        {
            list.Add(item.ValueKind == JsonValueKind.Number ? item.GetDouble() : null);
        }
        return list.AsReadOnly();
    }
}
