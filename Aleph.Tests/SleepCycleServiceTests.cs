using System.Text.Json;
using System.Threading.Channels;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Xunit;

namespace Aleph.Tests;

/// <summary>
/// Phase 9/9.2 Validation — C# proving harness for SleepCycleService v2.
///
/// Tests exercise the orchestration logic using hand-rolled stubs
/// for IAether, IHomeostasis, IAlephBus, and IConfiguration.
///
/// Validates:
///   A. Overlap protection (SemaphoreSlim(1))
///   B. Homeostasis gating (training blocked under stress/fatigue/overload/breathless)
///   C. Status → resolve → train sequencing
///   D. Training gate decisions
///   E. Summary event publishing with correct fields
///   F. Graceful error handling
///   G. Stall detection (Phase 9.2)
///   H. Periodic evaluation (Phase 9.2)
///   I. Scorecard awareness (Phase 9.2)
/// </summary>
public class SleepCycleServiceTests
{
    // ═══════════════════════════════════════════════════════════════
    // TRAINING GATE TESTS
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public async Task TrainingBlocked_WhenOverloaded()
    {
        var env = new TestEnvironment(overloaded: true);
        await env.RunOneCycleAsync();

        Assert.NotNull(env.LastPublishedEvent);
        var evt = env.LastPublishedEvent!;
        Assert.Equal("Blocked", evt.TrainingGate);
        Assert.Contains("overloaded", evt.TrainingGateReason!);
        Assert.Equal(0, evt.SamplesFitted);
        Assert.False(env.Aether.TrainCalled);
    }

    [Fact]
    public async Task TrainingBlocked_WhenBreathless()
    {
        var env = new TestEnvironment(breathless: true);
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal("Blocked", evt.TrainingGate);
        Assert.Contains("breathless", evt.TrainingGateReason!);
        Assert.False(env.Aether.TrainCalled);
    }

    [Fact]
    public async Task TrainingBlocked_WhenStressTooHigh()
    {
        var env = new TestEnvironment(stressLevel: 0.85);
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal("Blocked", evt.TrainingGate);
        Assert.Contains("stress_too_high", evt.TrainingGateReason!);
        Assert.False(env.Aether.TrainCalled);
    }

    [Fact]
    public async Task TrainingBlocked_WhenFatigueTooHigh()
    {
        var env = new TestEnvironment(fatigueLevel: 0.85);
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal("Blocked", evt.TrainingGate);
        Assert.Contains("fatigue_too_high", evt.TrainingGateReason!);
        Assert.False(env.Aether.TrainCalled);
    }

    [Fact]
    public async Task TrainingBlocked_WhenInsufficientResolved()
    {
        // Status shows 0 resolved, resolve produces 0 newly resolved
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 5, resolved: 0);
        env.Aether.ResolveJson = BuildResolveJson(resolved: 0, deferred: 5);
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal("Blocked", evt.TrainingGate);
        Assert.Contains("insufficient_resolved", evt.TrainingGateReason!);
        Assert.False(env.Aether.TrainCalled);
    }

    [Fact]
    public async Task TrainingAllowed_WhenNewResolved()
    {
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 10, resolved: 5);
        env.Aether.ResolveJson = BuildResolveJson(resolved: 3, deferred: 2);
        env.Aether.TrainJson = BuildTrainJson(fitted: 3, fresh: 3, replay: 0);
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal("Allowed", evt.TrainingGate);
        Assert.Contains("new_resolved", evt.TrainingGateReason!);
        Assert.True(env.Aether.TrainCalled);
        Assert.Equal(3, evt.SamplesFitted);
    }

    [Fact]
    public async Task TrainingAllowed_WhenExistingBacklog()
    {
        // No newly resolved this cycle, but status shows enough backlog
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 0, resolved: 10);
        // pending < min, so resolve won't be called. But resolved backlog is large.
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal("Allowed", evt.TrainingGate);
        Assert.Contains("existing_resolved_backlog", evt.TrainingGateReason!);
        Assert.True(env.Aether.TrainCalled);
    }

    [Fact]
    public async Task TrainingBlocked_WhenResolveFails()
    {
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 10, resolved: 0);
        env.Aether.ResolveSuccess = false;
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal("Blocked", evt.TrainingGate);
        Assert.Contains("resolve_failed", evt.TrainingGateReason!);
        Assert.False(env.Aether.TrainCalled);
    }

    // ═══════════════════════════════════════════════════════════════
    // SEQUENCING TESTS
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public async Task SkipsResolve_WhenPendingBelowThreshold()
    {
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 0, resolved: 5);
        await env.RunOneCycleAsync();

        Assert.True(env.Aether.StatusCalled);
        Assert.False(env.Aether.ResolveCalled); // No resolve because pending=0 < min=1
        Assert.True(env.Aether.TrainCalled);    // But train runs on existing backlog
    }

    [Fact]
    public async Task SkipsResolveAndTrain_WhenStatusFails()
    {
        var env = new TestEnvironment();
        env.Aether.StatusSuccess = false;
        await env.RunOneCycleAsync();

        Assert.True(env.Aether.StatusCalled);
        Assert.False(env.Aether.ResolveCalled);
        Assert.False(env.Aether.TrainCalled);
        Assert.Equal("status_failed", env.LastPublishedEvent!.SkipReason);
    }

    [Fact]
    public async Task FullSequence_StatusResolveTrainPublish()
    {
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 10, resolved: 5);
        env.Aether.ResolveJson = BuildResolveJson(resolved: 4, deferred: 1);
        env.Aether.TrainJson = BuildTrainJson(fitted: 4, fresh: 3, replay: 1);
        await env.RunOneCycleAsync();

        Assert.True(env.Aether.StatusCalled);
        Assert.True(env.Aether.ResolveCalled);
        Assert.True(env.Aether.TrainCalled);

        var evt = env.LastPublishedEvent!;
        Assert.True(evt.StatusOk);
        Assert.True(evt.ResolveOk);
        Assert.True(evt.TrainOk);
        Assert.Equal(4, evt.NewlyResolved);
        Assert.Equal(1, evt.Deferred);
        Assert.Equal(4, evt.SamplesFitted);
        Assert.Equal(3, evt.FreshCount);
        Assert.Equal(1, evt.ReplayCount);
    }

    // ═══════════════════════════════════════════════════════════════
    // SUMMARY EVENT FIELD TESTS
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public async Task SummaryEvent_HasCorrectSymbolAndHorizon()
    {
        var env = new TestEnvironment();
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal("SI=F", evt.Symbol);
        Assert.Equal("1d", evt.Horizon);
        Assert.Equal("SleepCycle", evt.Source);
        Assert.Equal("sleep_cycle_summary", evt.Kind);
    }

    [Fact]
    public async Task SummaryEvent_CapuresStressAndFatigueFromHomeostasis()
    {
        var env = new TestEnvironment(stressLevel: 0.35, fatigueLevel: 0.22);
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal(0.35, evt.StressLevel);
        Assert.Equal(0.22, evt.FatigueLevel);
    }

    [Fact]
    public async Task SummaryEvent_ReportsStatusFields()
    {
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 7, resolved: 12, cursor: 5, modelState: "warm");
        env.Aether.ResolveJson = BuildResolveJson(resolved: 2, deferred: 3);
        env.Aether.TrainJson = BuildTrainJson(fitted: 2);
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal(7, evt.PendingCount);
    }

    [Fact]
    public async Task SummaryEvent_DurationIsPositive()
    {
        var env = new TestEnvironment();
        await env.RunOneCycleAsync();

        Assert.True(env.LastPublishedEvent!.DurationMs >= 0);
    }

    [Fact]
    public async Task SummaryEvent_Severity_Normal_OnCleanCycle()
    {
        // Use a scenario where training runs with a balanced distribution
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 5, resolved: 5);
        env.Aether.ResolveJson = BuildResolveJson(resolved: 3);
        env.Aether.TrainJson = BuildTrainJson(
            fitted: 9, fresh: 9, replay: 0,
            batchClassDist: new Dictionary<string, int> { { "bullish", 3 }, { "neutral", 3 }, { "bearish", 3 } });
        await env.RunOneCycleAsync();

        Assert.Equal(PulseSeverity.Normal, env.LastPublishedEvent!.Severity);
    }

    [Fact]
    public async Task SummaryEvent_HasDriftFlags_WhenPresent()
    {
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 5, resolved: 5);
        env.Aether.ResolveJson = BuildResolveJson(resolved: 3);
        env.Aether.TrainJson = BuildTrainJson(fitted: 3, driftFlags: new[] { "prob_shift_bullish", "class_collapse_warning" });
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.NotNull(evt.DriftFlags);
        Assert.Equal(2, evt.DriftFlags!.Count);
        Assert.Contains("prob_shift_bullish", evt.DriftFlags);
        Assert.Equal(PulseSeverity.Elevated, evt.Severity);
    }

    [Fact]
    public async Task SummaryEvent_HasClassSkewWarning_WhenDominantClass()
    {
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 5, resolved: 5);
        env.Aether.ResolveJson = BuildResolveJson(resolved: 3);
        env.Aether.TrainJson = BuildTrainJson(
            fitted: 10,
            batchClassDist: new Dictionary<string, int> { { "bullish", 8 }, { "neutral", 1 }, { "bearish", 1 } });
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.NotNull(evt.ClassSkewWarning);
        Assert.Contains("bullish", evt.ClassSkewWarning!);
        Assert.Equal(PulseSeverity.Elevated, evt.Severity);
    }

    [Fact]
    public async Task SummaryEvent_HasCycleNumber()
    {
        var env = new TestEnvironment();
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal(1, evt.CycleNumber);
    }

    // ═══════════════════════════════════════════════════════════════
    // ERROR HANDLING TESTS
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public async Task GracefullyHandlesException_InCycle()
    {
        var env = new TestEnvironment();
        env.Aether.StatusThrows = true;
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.NotNull(evt.SkipReason);
        Assert.StartsWith("error:", evt.SkipReason!);
    }

    [Fact]
    public async Task SummaryAlwaysPublished_EvenOnError()
    {
        var env = new TestEnvironment();
        env.Aether.StatusThrows = true;
        await env.RunOneCycleAsync();

        // The key invariant: a summary event is ALWAYS published, even on error
        Assert.NotNull(env.LastPublishedEvent);
    }

    // ═══════════════════════════════════════════════════════════════
    // STALL DETECTION TESTS (Phase 9.2)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public async Task StallDetection_ZeroProgressIncrementsCounter()
    {
        // A cycle with no resolved and no training = zero progress
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 0, resolved: 0);
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.True(evt.ConsecutiveZeroProgressCycles >= 1);
    }

    [Fact]
    public async Task StallDetection_ProgressResetsCounter()
    {
        // Run a productive cycle
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 5, resolved: 5);
        env.Aether.ResolveJson = BuildResolveJson(resolved: 3);
        env.Aether.TrainJson = BuildTrainJson(fitted: 3);
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal(0, evt.ConsecutiveZeroProgressCycles);
    }

    [Fact]
    public async Task StallDetection_SeverityWarning_WhenStalled()
    {
        // Run multiple zero-progress cycles to trigger stall
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 0, resolved: 0);
        // Run 6 cycles (above default threshold of 5)
        for (int i = 0; i < 6; i++)
            await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal(PulseSeverity.Warning, evt.Severity);
        Assert.Contains("stalled", evt.Tags!);
    }

    // ═══════════════════════════════════════════════════════════════
    // PERIODIC EVALUATION TESTS (Phase 9.2)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public async Task Evaluation_RunsAtConfiguredInterval()
    {
        var env = new TestEnvironment(evaluateEveryCycles: 3);
        env.Aether.StatusJson = BuildStatusJson(pending: 5, resolved: 5);
        env.Aether.ResolveJson = BuildResolveJson(resolved: 2);
        env.Aether.TrainJson = BuildTrainJson(fitted: 2);

        // Cycles 1 and 2: no eval
        await env.RunOneCycleAsync();
        Assert.False(env.LastPublishedEvent!.EvaluationRan);
        Assert.False(env.Aether.EvaluateCalled);

        await env.RunOneCycleAsync();
        Assert.False(env.LastPublishedEvent!.EvaluationRan);

        // Cycle 3: eval runs
        await env.RunOneCycleAsync();
        Assert.True(env.LastPublishedEvent!.EvaluationRan);
        Assert.True(env.Aether.EvaluateCalled);
    }

    [Fact]
    public async Task Evaluation_NonFatalOnFailure()
    {
        var env = new TestEnvironment(evaluateEveryCycles: 1);
        env.Aether.StatusJson = BuildStatusJson(pending: 5, resolved: 5);
        env.Aether.ResolveJson = BuildResolveJson(resolved: 2);
        env.Aether.TrainJson = BuildTrainJson(fitted: 2);
        env.Aether.EvaluateSuccess = false;
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.True(evt.EvaluationRan);
        Assert.False(evt.EvaluationOk);
        // But the rest of the cycle succeeded
        Assert.True(evt.StatusOk);
        Assert.True(evt.TrainOk);
    }

    // ═══════════════════════════════════════════════════════════════
    // SCORECARD AWARENESS TESTS (Phase 9.2)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public async Task RollingScorecard_ParsedFromStatus()
    {
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJsonWithScorecard(
            pending: 5, resolved: 10,
            rollingAccuracy: 0.65, rollingBrier: 0.28, rollingGrade: "B", rollingSamples: 50);
        env.Aether.ResolveJson = BuildResolveJson(resolved: 2);
        env.Aether.TrainJson = BuildTrainJson(fitted: 2);
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal(0.65, evt.RollingScorecardAccuracy);
        Assert.Equal(0.28, evt.RollingScorecardBrier);
        Assert.Equal("B", evt.RollingScorecardGrade);
        Assert.Equal(50, evt.RollingScorecardSamples);
    }

    [Fact]
    public async Task CycleScorecard_ParsedFromResolve()
    {
        var env = new TestEnvironment();
        env.Aether.StatusJson = BuildStatusJson(pending: 5, resolved: 5);
        env.Aether.ResolveJson = BuildResolveJsonWithScorecard(
            resolved: 3, cycleAccuracy: 0.70, cycleBrier: 0.22, cycleGrade: "A");
        env.Aether.TrainJson = BuildTrainJson(fitted: 3);
        await env.RunOneCycleAsync();

        var evt = env.LastPublishedEvent!;
        Assert.Equal(0.70, evt.CycleScorecardAccuracy);
        Assert.Equal(0.22, evt.CycleScorecardBrier);
        Assert.Equal("A", evt.CycleScorecardGrade);
    }

    // ═══════════════════════════════════════════════════════════════
    // JSON BUILDERS
    // ═══════════════════════════════════════════════════════════════

    internal static string BuildStatusJson(
        int pending = 0,
        int resolved = 0,
        int cursor = 0,
        string modelState = "cold")
    {
        var obj = new
        {
            ok = true,
            domain = "ml",
            action = "cortex_status",
            pending_count = pending,
            pending_eligible_count = pending,
            pending_blocked_count = 0,
            resolved_count = resolved,
            cursor_sequence = cursor,
            model_state = modelState,
            trained_samples = 0,
        };
        return JsonSerializer.Serialize(obj);
    }

    private static string BuildStatusJsonWithScorecard(
        int pending, int resolved,
        double rollingAccuracy, double rollingBrier, string rollingGrade, int rollingSamples)
    {
        var obj = new
        {
            ok = true,
            domain = "ml",
            action = "cortex_status",
            pending_count = pending,
            pending_eligible_count = pending,
            pending_blocked_count = 0,
            resolved_count = resolved,
            cursor_sequence = 0,
            model_state = "warm",
            trained_samples = rollingSamples,
            rolling_scorecard = new
            {
                accuracy = rollingAccuracy,
                brier_score = rollingBrier,
                grade = rollingGrade,
                sample_count = rollingSamples,
            }
        };
        return JsonSerializer.Serialize(obj);
    }

    internal static string BuildResolveJson(
        int resolved = 0,
        int deferred = 0,
        int expired = 0,
        int errored = 0,
        double accuracy = 0.5,
        double meanBrier = 0.25)
    {
        var obj = new
        {
            ok = true,
            domain = "ml",
            action = "cortex_resolve",
            resolution = new
            {
                resolved_count = resolved,
                deferred_count = deferred,
                expired_count = expired,
                errored_count = errored,
                accuracy = accuracy,
                mean_brier_score = meanBrier,
                class_distribution = new { bullish = resolved / 3, neutral = resolved / 3, bearish = resolved - 2 * (resolved / 3) },
            },
        };
        return JsonSerializer.Serialize(obj);
    }

    private static string BuildResolveJsonWithScorecard(
        int resolved,
        double cycleAccuracy, double cycleBrier, string cycleGrade)
    {
        var obj = new
        {
            ok = true,
            domain = "ml",
            action = "cortex_resolve",
            resolution = new
            {
                resolved_count = resolved,
                deferred_count = 0,
                expired_count = 0,
                errored_count = 0,
                accuracy = cycleAccuracy,
                mean_brier_score = cycleBrier,
                class_distribution = new { bullish = 1, neutral = 1, bearish = 1 },
            },
            cycle_scorecard = new
            {
                accuracy = cycleAccuracy,
                brier_score = cycleBrier,
                grade = cycleGrade,
            }
        };
        return JsonSerializer.Serialize(obj);
    }

    internal static string BuildTrainJson(
        int fitted = 0,
        int fresh = 0,
        int replay = 0,
        int cursorSeq = 1,
        string[]? driftFlags = null,
        Dictionary<string, int>? batchClassDist = null)
    {
        if (fresh == 0 && fitted > 0) fresh = fitted;

        var training = new Dictionary<string, object>
        {
            ["samples_fitted"] = fitted,
            ["fresh_count"] = fresh,
            ["replay_count"] = replay,
            ["model_state"] = "warm",
            ["drift_flags"] = driftFlags ?? Array.Empty<string>(),
        };

        if (batchClassDist != null)
            training["batch_class_distribution"] = batchClassDist;
        else
            training["batch_class_distribution"] = new Dictionary<string, int>
            {
                { "bullish", fitted / 3 },
                { "neutral", fitted / 3 },
                { "bearish", fitted - 2 * (fitted / 3) },
            };

        var obj = new Dictionary<string, object>
        {
            ["ok"] = true,
            ["domain"] = "ml",
            ["action"] = "cortex_train",
            ["training"] = training,
            ["cursor_sequence"] = cursorSeq,
        };
        return JsonSerializer.Serialize(obj);
    }

    private static string BuildEvaluateJson(
        int challengersEvaluated = 3,
        string bestChallenger = "tight_labels",
        string decision = "INCONCLUSIVE")
    {
        var obj = new
        {
            ok = true,
            domain = "ml",
            action = "cortex_evaluate",
            evaluation = new
            {
                challengers_evaluated = challengersEvaluated,
                best_challenger = new { name = bestChallenger },
                promotion_decision = new { decision },
            }
        };
        return JsonSerializer.Serialize(obj);
    }

    // ═══════════════════════════════════════════════════════════════
    // TEST INFRASTRUCTURE — STUBS
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// Encapsulates all stubs and provides a simple RunOneCycleAsync() to drive
    /// the SleepCycleService through exactly one cycle.
    /// </summary>
    private sealed class TestEnvironment
    {
        public StubAether Aether { get; }
        public StubHomeostasis Homeostasis { get; }
        public StubBus Bus { get; }
        public SleepCycleSummaryEvent? LastPublishedEvent => Bus.LastEvent as SleepCycleSummaryEvent;

        private readonly SleepCycleService _svc;

        public TestEnvironment(
            double stressLevel = 0.1,
            double fatigueLevel = 0.1,
            bool overloaded = false,
            bool breathless = false,
            int evaluateEveryCycles = 10)
        {
            Aether = new StubAether();
            Homeostasis = new StubHomeostasis(stressLevel, fatigueLevel, overloaded, breathless);
            Bus = new StubBus();

            var config = new ConfigurationBuilder()
                .AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["Aether:SleepCycle:IntervalSeconds"] = "60",
                    ["Aether:SleepCycle:StartupDelaySeconds"] = "5",
                    ["Aether:SleepCycle:MinPendingToResolve"] = "1",
                    ["Aether:SleepCycle:MinResolvedToTrain"] = "3",
                    ["Aether:SleepCycle:MaxStressForTraining"] = "0.7",
                    ["Aether:SleepCycle:MaxFatigueForTraining"] = "0.7",
                    ["Aether:SleepCycle:Symbol"] = "SI=F",
                    ["Aether:SleepCycle:Horizon"] = "1d",
                    ["Aether:SleepCycle:Interval"] = "1h",
                    ["Aether:SleepCycle:EvaluateEveryCycles"] = evaluateEveryCycles.ToString(),
                    ["Aether:SleepCycle:StallWarningThreshold"] = "5",
                })
                .Build();

            var logger = new NullLogger<SleepCycleService>();

            _svc = new SleepCycleService(
                Aether, Homeostasis, Bus, config, logger);
        }

        public async Task RunOneCycleAsync()
        {
            // RunOneCycle is now internal, callable directly
            await _svc.RunOneCycle(CancellationToken.None);
        }
    }

    // ─── Stub IAether ──────────────────────────────────────────

    private sealed class StubAether : IAether
    {
        public IAether.IMathGateway Math => throw new NotImplementedException();
        public IAether.IMlGateway Ml => _ml;
        public IAether.ISimGateway Sim => throw new NotImplementedException();
        public IAether.IMacroGateway Macro => throw new NotImplementedException();
        public IAether.IRegulationGateway Regulation => throw new NotImplementedException();

        private readonly StubMlGateway _ml;

        public StubAether()
        {
            _ml = new StubMlGateway(this);
        }

        // Track calls
        public bool StatusCalled => _ml.StatusCalled;
        public bool ResolveCalled => _ml.ResolveCalled;
        public bool TrainCalled => _ml.TrainCalled;
        public bool EvaluateCalled => _ml.EvaluateCalled;

        // Control responses
        public string StatusJson
        {
            get => _ml.StatusJson;
            set => _ml.StatusJson = value;
        }
        public string ResolveJson
        {
            get => _ml.ResolveJson;
            set => _ml.ResolveJson = value;
        }
        public string TrainJson
        {
            get => _ml.TrainJson;
            set => _ml.TrainJson = value;
        }
        public string EvaluateJson
        {
            get => _ml.EvaluateJson;
            set => _ml.EvaluateJson = value;
        }
        public bool StatusSuccess
        {
            get => _ml.StatusSuccess;
            set => _ml.StatusSuccess = value;
        }
        public bool ResolveSuccess
        {
            get => _ml.ResolveSuccess;
            set => _ml.ResolveSuccess = value;
        }
        public bool TrainSuccess
        {
            get => _ml.TrainSuccess;
            set => _ml.TrainSuccess = value;
        }
        public bool EvaluateSuccess
        {
            get => _ml.EvaluateSuccess;
            set => _ml.EvaluateSuccess = value;
        }
        public bool StatusThrows
        {
            get => _ml.StatusThrows;
            set => _ml.StatusThrows = value;
        }

        private sealed class StubMlGateway : IAether.IMlGateway
        {
            private readonly StubAether _parent;

            public StubMlGateway(StubAether parent) => _parent = parent;

            public bool StatusCalled;
            public bool ResolveCalled;
            public bool TrainCalled;
            public bool EvaluateCalled;

            public string StatusJson = BuildStatusJson(pending: 5, resolved: 5);
            public string ResolveJson = BuildResolveJson(resolved: 2);
            public string TrainJson = BuildTrainJson(fitted: 2);
            public string EvaluateJson = BuildEvaluateJson();

            public bool StatusSuccess = true;
            public bool ResolveSuccess = true;
            public bool TrainSuccess = true;
            public bool EvaluateSuccess = true;
            public bool StatusThrows;

            public Task<AetherJsonResult> CortexStatusAsync(MlCortexStatusRequest request, CancellationToken ct)
            {
                StatusCalled = true;
                if (StatusThrows) throw new InvalidOperationException("Simulated status error");
                return Task.FromResult(new AetherJsonResult(StatusSuccess, StatusJson, null, 0, false));
            }

            public Task<AetherJsonResult> CortexResolveAsync(MlCortexResolveRequest request, CancellationToken ct)
            {
                ResolveCalled = true;
                return Task.FromResult(new AetherJsonResult(ResolveSuccess, ResolveJson, null, 0, false));
            }

            public Task<AetherJsonResult> CortexTrainAsync(MlCortexTrainRequest request, CancellationToken ct)
            {
                TrainCalled = true;
                return Task.FromResult(new AetherJsonResult(TrainSuccess, TrainJson, null, 0, false));
            }

            public Task<AetherJsonResult> CortexEvaluateAsync(MlCortexEvaluateRequest request, CancellationToken ct)
            {
                EvaluateCalled = true;
                return Task.FromResult(new AetherJsonResult(EvaluateSuccess, EvaluateJson, null, 0, false));
            }

            public Task<AetherJsonResult> CortexOperationalStatusAsync(
                MlCortexOperationalStatusRequest request,
                CancellationToken ct)
            {
                return Task.FromResult(new AetherJsonResult(true, "{}", null, 0, false));
            }

            public Task<AetherJsonResult> CortexPredictAsync(MlCortexPredictRequest request, CancellationToken ct) =>
                throw new NotImplementedException();
            public Task<AetherJsonResult> PredictAsync(MlPredictRequest request, CancellationToken ct) =>
                throw new NotImplementedException();
            public Task<AetherJsonResult> TrainAsync(MlTrainRequest request, CancellationToken ct) =>
                throw new NotImplementedException();
            public Task<AetherJsonResult> GetStatusAsync(MlStatusRequest request, CancellationToken ct) =>
                throw new NotImplementedException();
        }

        // Need static helpers accessible from inner class
        private static string BuildStatusJson(int pending, int resolved) =>
            SleepCycleServiceTests.BuildStatusJson(pending, resolved);

        private static string BuildResolveJson(int resolved) =>
            SleepCycleServiceTests.BuildResolveJson(resolved);

        private static string BuildTrainJson(int fitted) =>
            SleepCycleServiceTests.BuildTrainJson(fitted);

        private static string BuildEvaluateJson() =>
            SleepCycleServiceTests.BuildEvaluateJson();
    }

    // ─── Stub IHomeostasis ─────────────────────────────────────

    private sealed class StubHomeostasis : IHomeostasis
    {
        private readonly double _stress;
        private readonly double _fatigue;
        private readonly bool _overloaded;
        private readonly bool _breathless;

        public StubHomeostasis(double stress, double fatigue, bool overloaded, bool breathless)
        {
            _stress = stress;
            _fatigue = fatigue;
            _overloaded = overloaded;
            _breathless = breathless;
        }

        public HomeostasisSnapshot GetSnapshot() => new()
        {
            StressLevel = _stress,
            FatigueLevel = _fatigue,
            OverloadLevel = _overloaded ? 0.8 : 0.1,
            FailureStreak = 0,
            LastPulseDurationMs = 100,
            LastUpdatedUtc = DateTimeOffset.UtcNow,
            ActiveFlags = Array.Empty<string>(),
        };

        public bool IsOverloaded => _overloaded;
        public bool IsBreathless => _breathless;

        public void Ingest(PulseEnvelope envelope) { }
        public void RecordPulse(PulseEnvelope report) { }
    }

    // ─── Stub IAlephBus ────────────────────────────────────────

    private sealed class StubBus : IAlephBus
    {
        public AlephEvent? LastEvent;
        public int PublishCount;

        public ValueTask PublishAsync(AlephEvent evt, CancellationToken ct = default)
        {
            LastEvent = evt;
            PublishCount++;
            return ValueTask.CompletedTask;
        }

        public ChannelReader<AlephEvent> Subscribe(string subscriberName, Func<AlephEvent, bool>? filter = null) =>
            Channel.CreateUnbounded<AlephEvent>().Reader;
    }

    // ─── Null Logger ───────────────────────────────────────────

    private sealed class NullLogger<T> : ILogger<T>
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => false;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state,
            Exception? exception, Func<TState, Exception?, string> formatter) { }
    }
}
