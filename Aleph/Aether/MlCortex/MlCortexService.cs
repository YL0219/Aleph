using System.Text.Json;

namespace Aleph;

/// <summary>
/// The ML Cortex — predictive organ of the Aleph organism.
///
/// Subscribes to the AlephBus for MetabolicEvent blood cells, passes them
/// through the Python sandbox brain for prediction, and publishes refined
/// PredictionEvent blood cells back into the bloodstream.
///
/// Design principles:
///   - C# is thin plumbing only — all ML logic lives in Python
///   - Real-time inference path is always available (even cold start)
///   - Training is NOT performed here; it lives in the Sleep Cycle path
///   - Multi-horizon-ready architecture, v1 uses a single configured horizon
///   - Hot-swappable: Python brain can be replaced without C# changes
///   - Temporal security enforced: unsafe samples blocked from training memory
/// </summary>
public sealed class MlCortexService : BackgroundService, IAlephOrgan
{
    private const int PredictionTimeoutMs = 30_000;
    private const string DefaultModelKey = "cortex_sgd_1h_24bar";
    private const string DefaultFeatureVersion = "v2.0.0";
    private const string DefaultTemporalPolicyVersion = "tp_v1";
    private const int DefaultHorizonBars = 24;
    private const string DefaultInterval = "1h";

    private readonly IAlephBus _bus;
    private readonly IAether _aether;
    private readonly IHomeostasis _homeostasis;
    private readonly IConfiguration _configuration;
    private readonly ILogger<MlCortexService> _logger;

    private volatile bool _isActive;

    public string OrganName => "MlCortex";

    public IReadOnlyList<string> EventInterests { get; } =
        new[] { nameof(MetabolicEvent) }.AsReadOnly();

    public bool IsActive => _isActive;

    /// <summary>The active horizon key for v1. Configurable, defaults to "1d".</summary>
    private string ActiveHorizon { get; }

    private string ModelKey { get; }
    private string FeatureVersion { get; }
    private int HorizonBars { get; }

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
    };

    public MlCortexService(
        IAlephBus bus,
        IAether aether,
        IHomeostasis homeostasis,
        IConfiguration configuration,
        ILogger<MlCortexService> logger)
    {
        _bus = bus;
        _aether = aether;
        _homeostasis = homeostasis;
        _configuration = configuration;
        _logger = logger;

        ActiveHorizon = configuration["Aether:Cortex:ActiveHorizon"] ?? "1d";
        ModelKey = configuration["Aether:Cortex:ModelKey"] ?? DefaultModelKey;
        FeatureVersion = configuration["Aether:Cortex:FeatureVersion"] ?? DefaultFeatureVersion;
        HorizonBars = int.TryParse(configuration["Aether:Cortex:HorizonBars"], out var hb) ? hb : DefaultHorizonBars;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogDebug("[MlCortex] Predictive organ starting. Subscribing to bloodstream...");

        var reader = _bus.Subscribe(
            OrganName,
            static evt => evt is MetabolicEvent);

        _isActive = true;

        _logger.LogDebug("[MlCortex] Active. Horizon={Horizon}. Awaiting MetabolicEvent blood cells.", ActiveHorizon);

        try
        {
            await foreach (var evt in reader.ReadAllAsync(stoppingToken))
            {
                if (evt is not MetabolicEvent me)
                    continue;

                try
                {
                    await ProcessMetabolicEventAsync(me, stoppingToken);
                }
                catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex,
                        "[MlCortex] Unhandled error processing {Symbol}/{Interval}.",
                        me.Symbol, me.Interval);
                }
            }
        }
        catch (OperationCanceledException) { /* expected on shutdown */ }
        finally
        {
            _isActive = false;
            _logger.LogInformation("[MlCortex] Predictive organ stopped.");
        }
    }

    /// <summary>
    /// Process a single MetabolicEvent: build payload → call Python brain → publish PredictionEvent.
    /// </summary>
    private async Task ProcessMetabolicEventAsync(MetabolicEvent me, CancellationToken ct)
    {
        _logger.LogDebug(
            "[MlCortex] Processing {Symbol}/{Interval} (metabolic event {EventId}).",
            me.Symbol, me.Interval, me.EventId);

        // ── Step 1: Determine temporal safety and governance ──
        var temporal = me.Temporal;
        var pointInTimeSafe = temporal?.PointInTimeSafe ?? true; // default safe if no temporal envelope yet
        var observationCutoffUtc = temporal?.ObservationCutoffUtc ?? me.AsOfUtc;

        var snapshot = _homeostasis.GetSnapshot();
        var isOverloaded = _homeostasis.IsOverloaded;
        var isBreathless = _homeostasis.IsBreathless;

        // Governance: determine training eligibility
        var learningBlockReasons = new List<string>();
        if (!pointInTimeSafe)
            learningBlockReasons.Add("temporal_safety_failed");
        if (isOverloaded)
            learningBlockReasons.Add("system_overloaded");
        if (isBreathless)
            learningBlockReasons.Add("system_breathless");

        var eligibleForTraining = learningBlockReasons.Count == 0;

        // ── Step 2: Build the nested ML input payload ──
        var metabolicPayloadJson = BuildMetabolicPayload(me, snapshot, isOverloaded, isBreathless,
            eligibleForTraining, learningBlockReasons);

        // ── Step 3: Call IAether.Ml.CortexPredictAsync ──
        var request = new MlCortexPredictRequest
        {
            Symbol = me.Symbol,
            Interval = DefaultInterval,
            ActiveHorizon = ActiveHorizon,
            AsOfUtc = me.AsOfUtc,
            MetabolicPayloadJson = metabolicPayloadJson
        };

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(PredictionTimeoutMs);

        AetherJsonResult result;
        try
        {
            result = await _aether.Ml.CortexPredictAsync(request, timeoutCts.Token);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            _logger.LogWarning(
                "[MlCortex] Prediction timed out for {Symbol}/{Interval} after {Timeout}ms.",
                me.Symbol, me.Interval, PredictionTimeoutMs);
            return;
        }

        if (!result.Success || string.IsNullOrWhiteSpace(result.PayloadJson))
        {
            _logger.LogWarning(
                "[MlCortex] Python brain failed for {Symbol}/{Interval}: {Error}",
                me.Symbol, me.Interval, result.Error ?? "empty payload");
            return;
        }

        // ── Step 4: Parse Python brain output ──
        PredictionEvent? predictionEvent;
        try
        {
            predictionEvent = ParsePredictionOutput(me, result.PayloadJson, observationCutoffUtc);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex,
                "[MlCortex] Failed to parse prediction JSON for {Symbol}/{Interval}.",
                me.Symbol, me.Interval);
            return;
        }

        if (predictionEvent is null)
            return;

        // ── Step 5: Publish PredictionEvent into the bloodstream ──
        await _bus.PublishAsync(predictionEvent, ct);

        _logger.LogInformation(
            "[MlCortex] {Symbol}/{Interval} → {Class} Conf={Confidence:F3} Tend={Tendency:F3} ({State}) Safe={Safe}",
            me.Symbol, me.Interval,
            predictionEvent.PredictedClass, predictionEvent.Confidence,
            predictionEvent.ActionTendency, predictionEvent.ModelState,
            predictionEvent.TemporalSecurityPassed);
    }

    // ═════════════════════════════════════════════════════════════════
    // Payload Builder — nested JSON contract for Python brain
    // ═════════════════════════════════════════════════════════════════

    private string BuildMetabolicPayload(
        MetabolicEvent me,
        HomeostasisSnapshot snapshot,
        bool isOverloaded,
        bool isBreathless,
        bool eligibleForTraining,
        List<string> learningBlockReasons)
    {
        var temporal = me.Temporal;
        var macro = me.MacroContext;

        var payload = new Dictionary<string, object?>
        {
            // ── meta ──
            ["meta"] = new Dictionary<string, object?>
            {
                ["symbol"] = me.Symbol,
                ["interval"] = me.Interval,
                ["asof_utc"] = me.AsOfUtc,
                ["source_event_id"] = me.EventId.ToString(),
                ["metabolic_version"] = me.MetabolicVersion,
                ["model_key"] = ModelKey,
                ["feature_version"] = FeatureVersion,
                ["active_horizon"] = ActiveHorizon,
                ["horizon_bars"] = HorizonBars,
            },

            // ── temporal ──
            ["temporal"] = new Dictionary<string, object?>
            {
                ["bar_open_utc"] = temporal?.BarOpenUtc,
                ["bar_close_utc"] = temporal?.BarCloseUtc,
                ["observation_cutoff_utc"] = temporal?.ObservationCutoffUtc ?? me.AsOfUtc,
                ["max_included_knowledge_utc"] = temporal?.MaxIncludedKnowledgeUtc,
                ["point_in_time_safe"] = temporal?.PointInTimeSafe ?? true,
                ["temporal_policy_version"] = temporal?.TemporalPolicyVersion ?? DefaultTemporalPolicyVersion,
                ["historical_replay_mode"] = temporal?.HistoricalReplayMode ?? false,
                ["exclusion_reasons"] = temporal?.ExclusionReasons ?? (IReadOnlyList<string>)Array.Empty<string>(),
            },

            // ── technical ──
            ["technical"] = BuildTechnicalSection(me),

            // ── macro ──
            ["macro"] = BuildMacroSection(macro),

            // ── events ──
            ["events"] = BuildEventsSection(macro),

            // ── homeostasis ──
            ["homeostasis"] = new Dictionary<string, object?>
            {
                ["stress_level"] = snapshot.StressLevel,
                ["fatigue_level"] = snapshot.FatigueLevel,
                ["is_overloaded"] = isOverloaded,
                ["is_breathless"] = isBreathless,
            },

            // ── governance ──
            ["governance"] = new Dictionary<string, object?>
            {
                ["eligible_for_prediction"] = true,
                ["eligible_for_training"] = eligibleForTraining,
                ["learning_block_reasons"] = learningBlockReasons,
            },
        };

        return JsonSerializer.Serialize(payload, JsonOpts);
    }

    private static Dictionary<string, object?> BuildTechnicalSection(MetabolicEvent me)
    {
        var snap = me.Snapshot;
        var fs = me.FactorScores;
        var comp = me.Composite;

        var section = new Dictionary<string, object?>
        {
            ["bias"] = me.Bias,
            ["confidence"] = me.Confidence,
            ["row_count"] = me.RowCount,
            ["enough_for_long_trend"] = me.EnoughForLongTrend,
        };

        // Snapshot
        if (snap is not null)
        {
            section["price"] = snap.Price;
            section["sma_20"] = snap.Sma20;
            section["sma_50"] = snap.Sma50;
            section["sma_200"] = snap.Sma200;
            section["ema_12"] = snap.Ema12;
            section["ema_26"] = snap.Ema26;
            section["rsi_14"] = snap.Rsi14;
            section["atr_14"] = snap.Atr14;
            section["atr_pct"] = snap.AtrPct;
            section["volatility_20"] = snap.Volatility20;
            section["volume_sma_20"] = snap.VolumeSma20;
            section["dist_sma_20"] = snap.DistSma20;
            section["dist_sma_50"] = snap.DistSma50;
            section["dist_sma_200"] = snap.DistSma200;

            if (snap.Macd is not null)
            {
                section["macd_line"] = snap.Macd.Line;
                section["macd_signal"] = snap.Macd.Signal;
                section["macd_histogram"] = snap.Macd.Histogram;
            }

            if (snap.Bollinger is not null)
            {
                section["bb_bandwidth"] = snap.Bollinger.Bandwidth;
            }
        }

        // Factor scores grouped
        if (fs is not null)
        {
            section["factors"] = new Dictionary<string, object?>
            {
                ["trend"] = fs.Trend.Score,
                ["momentum"] = fs.Momentum.Score,
                ["volatility"] = fs.Volatility.Score,
                ["participation"] = fs.Participation.Score,
            };
        }

        // Composite probabilities grouped
        if (comp is not null)
        {
            section["composite"] = new Dictionary<string, object?>
            {
                ["bullish"] = comp.BullishProbability,
                ["bearish"] = comp.BearishProbability,
                ["neutral"] = comp.NeutralProbability,
                ["confidence"] = comp.Confidence,
            };
        }

        return section;
    }

    /// <summary>
    /// Build the macro section for the Python payload.
    ///
    /// New design (Phase 10.5): C# passes raw perception data as named sections.
    /// Python interprets the payload contents. To add new sections, update the
    /// perception pipeline and PerceptionSnapshotCache — this method adapts automatically.
    /// </summary>
    private static Dictionary<string, object?> BuildMacroSection(MetabolicMacroContext? macro)
    {
        var section = new Dictionary<string, object?>();

        if (macro is null)
            return section;

        // Envelope metadata — Python can use this for governance/freshness checks
        section["_meta"] = new Dictionary<string, object?>
        {
            ["snapshot_at_utc"] = macro.SnapshotAtUtc,
            ["freshness"] = macro.Freshness,
            ["sections_available"] = macro.SectionsAvailable,
            ["any_stale"] = macro.AnyStale,
            ["manifest_present"] = macro.ManifestPresent,
        };

        // Pass each section's raw payload through.
        // C# is opaque transport — Python interprets the JSON inside.
        foreach (var (name, sect) in macro.Sections)
        {
            if (string.IsNullOrEmpty(sect.PayloadJson))
            {
                // Section exists but has no data — include status only
                section[name] = new Dictionary<string, object?>
                {
                    ["_status"] = sect.Status,
                };
                continue;
            }

            try
            {
                // Deserialize the raw JSON so it embeds naturally (no double-escaping)
                var payload = JsonSerializer.Deserialize<JsonElement>(sect.PayloadJson);

                // Wrap in a dict with status metadata + the raw payload
                section[name] = new Dictionary<string, object?>
                {
                    ["_status"] = sect.Status,
                    ["_fetched_at_utc"] = sect.FetchedAtUtc,
                    ["_provider"] = sect.Provider,
                    ["data"] = payload,
                };
            }
            catch
            {
                section[name] = new Dictionary<string, object?>
                {
                    ["_status"] = "error",
                    ["_error"] = "payload_parse_failed",
                };
            }
        }

        return section;
    }

    /// <summary>
    /// Build the events section for the Python payload.
    ///
    /// With Phase 10.5, raw calendar and headline data flows through the macro
    /// section. This events section is kept for backwards compatibility — downstream
    /// feature extraction that reads events.materiality etc. gets safe 0.0 defaults.
    /// Future feature adapters should read from macro.calendar and macro.headlines instead.
    /// </summary>
    private static Dictionary<string, object?> BuildEventsSection(MetabolicMacroContext? macro)
    {
        // Backwards-compatible empty section.
        // Old feature_adapter paths (events.materiality, events.shock, etc.)
        // will find None → 0.0 defaults. Raw data is available in macro.calendar
        // and macro.headlines for future feature extraction.
        return new Dictionary<string, object?>();
    }

    // ═════════════════════════════════════════════════════════════════
    // Prediction Parser — reads expanded Python output
    // ═════════════════════════════════════════════════════════════════

    private PredictionEvent? ParsePredictionOutput(MetabolicEvent source, string payloadJson, string observationCutoffUtc)
    {
        using var doc = JsonDocument.Parse(payloadJson);
        var root = doc.RootElement;

        if (!root.TryGetProperty("ok", out var okProp) || !okProp.GetBoolean())
        {
            var error = root.TryGetProperty("error", out var errProp)
                ? errProp.GetString() ?? "unknown"
                : "unknown";
            _logger.LogWarning("[MlCortex] Python brain returned ok=false: {Error}", error);
            return null;
        }

        // Core fields
        var predictionId = SafeStr(root, "prediction_id") ?? Guid.NewGuid().ToString("N");
        var predClass = SafeStr(root, "predicted_class") ?? "neutral";
        var modelState = SafeStr(root, "model_state") ?? "cold_start";
        var modelVersion = SafeStr(root, "model_version") ?? "v1.0.0";
        var modelKey = SafeStr(root, "model_key") ?? ModelKey;
        var featureVersion = SafeStr(root, "feature_version") ?? FeatureVersion;
        var confidence = SafeDbl(root, "confidence");
        var actionTendency = SafeDbl(root, "action_tendency");
        var trainedSamples = SafeInt(root, "trained_samples");
        var pendingStored = SafeBool(root, "pending_sample_stored");
        var trainingOccurred = SafeBool(root, "training_occurred");
        var temporalSecurityPassed = SafeBool(root, "temporal_security_passed", fallback: true);
        var eligibleForTraining = SafeBool(root, "eligible_for_training") && temporalSecurityPassed;
        var priorityScore = SafeNullDbl(root, "priority_score");

        // Probabilities
        double pBullish = 0.33, pNeutral = 0.34, pBearish = 0.33;
        if (root.TryGetProperty("probabilities", out var probEl) && probEl.ValueKind == JsonValueKind.Object)
        {
            pBullish = SafeDbl(probEl, "bullish", 0.33);
            pNeutral = SafeDbl(probEl, "neutral", 0.34);
            pBearish = SafeDbl(probEl, "bearish", 0.33);
        }

        // Regime probabilities
        PredictionRegimeProbabilities? regimeProbs = null;
        if (root.TryGetProperty("regime_probabilities", out var rpEl) && rpEl.ValueKind == JsonValueKind.Object)
        {
            regimeProbs = new PredictionRegimeProbabilities
            {
                RiskOn = SafeDbl(rpEl, "risk_on"),
                RiskOff = SafeDbl(rpEl, "risk_off"),
                InflationPressure = SafeDbl(rpEl, "inflation_pressure"),
                GrowthScare = SafeDbl(rpEl, "growth_scare"),
                PolicyShock = SafeDbl(rpEl, "policy_shock"),
                FlightToSafety = SafeDbl(rpEl, "flight_to_safety"),
            };
        }

        // Event probabilities
        PredictionEventProbabilities? eventProbs = null;
        if (root.TryGetProperty("event_probabilities", out var epEl) && epEl.ValueKind == JsonValueKind.Object)
        {
            eventProbs = new PredictionEventProbabilities
            {
                Materiality = SafeDbl(epEl, "materiality"),
                FollowThrough = SafeDbl(epEl, "follow_through"),
                VolatilityExpansion = SafeDbl(epEl, "volatility_expansion"),
            };
        }

        // Top drivers / risks
        var topDrivers = ParseStringArray(root, "top_drivers");
        var topRisks = ParseStringArray(root, "top_risks");

        // Watched catalysts
        var watchedCatalysts = new List<PredictionCatalystRef>();
        if (root.TryGetProperty("watched_catalysts", out var wcArr) && wcArr.ValueKind == JsonValueKind.Array)
        {
            foreach (var wc in wcArr.EnumerateArray())
            {
                if (wc.ValueKind != JsonValueKind.Object) continue;
                var et = SafeStr(wc, "event_type");
                var sf = SafeStr(wc, "scheduled_for_utc");
                if (et is null || sf is null) continue;
                watchedCatalysts.Add(new PredictionCatalystRef
                {
                    EventType = et,
                    ScheduledForUtc = sf,
                    ImportanceProbability = SafeDbl(wc, "importance_probability"),
                });
            }
        }

        // Warnings
        var warnings = ParseStringArray(root, "warnings");

        return new PredictionEvent
        {
            OccurredAtUtc = DateTimeOffset.UtcNow,
            Source = "MlCortex",
            Kind = "cortex_prediction",
            Severity = PulseSeverity.Normal,
            Tags = new[] { "ml", "prediction", modelState },
            CorrelationId = source.CorrelationId,
            CausationId = source.EventId,

            PredictionId = predictionId,
            Symbol = source.Symbol,
            Interval = source.Interval,
            ActiveHorizon = ActiveHorizon,
            AsOfUtc = source.AsOfUtc,
            SourceMetabolicEventId = source.EventId,
            ModelVersion = modelVersion,
            ModelKey = modelKey,
            FeatureVersion = featureVersion,

            SourceObservationCutoffUtc = observationCutoffUtc,
            TemporalSecurityPassed = temporalSecurityPassed,
            EligibleForTraining = eligibleForTraining,

            ModelState = modelState,
            TrainedSamples = trainedSamples,

            PredictedClass = predClass,
            Probabilities = new PredictionProbabilities
            {
                Bullish = pBullish,
                Neutral = pNeutral,
                Bearish = pBearish,
            },
            Confidence = confidence,
            ActionTendency = actionTendency,

            RegimeProbabilities = regimeProbs,
            EventProbabilities = eventProbs,
            PriorityScore = priorityScore,
            TopDrivers = topDrivers,
            TopRisks = topRisks,
            WatchedCatalysts = watchedCatalysts.AsReadOnly(),

            PendingSampleStored = pendingStored,
            TrainingOccurred = trainingOccurred,
            Warnings = warnings,
        };
    }

    // ─── JSON helpers ────────────────────────────────────────────────

    private static string? SafeStr(JsonElement el, string prop)
    {
        return el.TryGetProperty(prop, out var v) && v.ValueKind == JsonValueKind.String
            ? v.GetString()
            : null;
    }

    private static double SafeDbl(JsonElement el, string prop, double fallback = 0)
    {
        return el.TryGetProperty(prop, out var v) && v.ValueKind == JsonValueKind.Number
            ? v.GetDouble()
            : fallback;
    }

    private static double? SafeNullDbl(JsonElement el, string prop)
    {
        return el.TryGetProperty(prop, out var v) && v.ValueKind == JsonValueKind.Number
            ? v.GetDouble()
            : null;
    }

    private static int SafeInt(JsonElement el, string prop)
    {
        return el.TryGetProperty(prop, out var v) && v.ValueKind == JsonValueKind.Number
            ? v.GetInt32()
            : 0;
    }

    private static bool SafeBool(JsonElement el, string prop, bool fallback = false)
    {
        return el.TryGetProperty(prop, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? v.GetBoolean()
            : fallback;
    }

    private static IReadOnlyList<string> ParseStringArray(JsonElement el, string prop)
    {
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
}
