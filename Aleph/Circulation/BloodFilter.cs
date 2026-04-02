// CONTRACT / INVARIANTS
// - Static validation methods for all blood cell types flowing through the AlephBus.
// - Acts as the "White Blood Cell" immune system — catches corrupted, poisoned, or
//   structurally invalid events before they reach organ processing logic.
// - NEVER modifies events. Returns (isHealthy, reason) tuples — caller decides.
// - Thread-safe: pure functions with no shared state.
// - Every organ that consumes AlephBus events MUST call the appropriate filter
//   before processing. Unhealthy events go to Quarantine, not to the pipeline.

namespace Aleph;

/// <summary>
/// Diagnostic result from a blood cell screening.
/// </summary>
public sealed record BloodScreening(bool IsHealthy, string? Anomaly = null)
{
    public static readonly BloodScreening Healthy = new(true);

    public static BloodScreening Reject(string reason) => new(false, reason);
}

/// <summary>
/// The Immune System — validates blood cells flowing through the AlephBus.
/// Organs must screen incoming events before processing them.
/// </summary>
public static class BloodFilter
{
    // ── MarketDataEvent screening (consumed by Liver) ──

    public static BloodScreening Screen(MarketDataEvent mde)
    {
        if (mde is null)
            return BloodScreening.Reject("Event is null.");

        if (string.IsNullOrWhiteSpace(mde.Symbol))
            return BloodScreening.Reject("Symbol is empty.");

        if (!SymbolValidator.IsValid(mde.Symbol))
            return BloodScreening.Reject($"Symbol '{mde.Symbol}' failed validation.");

        if (!mde.Success)
            return BloodScreening.Reject($"MarketDataEvent reported failure for {mde.Symbol}.");

        if (string.IsNullOrWhiteSpace(mde.Interval))
            return BloodScreening.Reject($"Interval is empty for {mde.Symbol}.");

        if (string.IsNullOrWhiteSpace(mde.SourceKind))
            return BloodScreening.Reject($"SourceKind is empty for {mde.Symbol}.");

        if (mde.OccurredAtUtc == default)
            return BloodScreening.Reject($"OccurredAtUtc is default for {mde.Symbol}.");

        return BloodScreening.Healthy;
    }

    // ── MetabolicEvent screening (consumed by MlCortex) ──

    public static BloodScreening Screen(MetabolicEvent me)
    {
        if (me is null)
            return BloodScreening.Reject("Event is null.");

        if (string.IsNullOrWhiteSpace(me.Symbol))
            return BloodScreening.Reject("Symbol is empty.");

        if (!SymbolValidator.IsValid(me.Symbol))
            return BloodScreening.Reject($"Symbol '{me.Symbol}' failed validation.");

        if (string.IsNullOrWhiteSpace(me.Interval))
            return BloodScreening.Reject($"Interval is empty for {me.Symbol}.");

        if (me.RowCount < 0)
            return BloodScreening.Reject($"Negative row count ({me.RowCount}) for {me.Symbol}.");

        if (me.Snapshot is null)
            return BloodScreening.Reject($"Snapshot is null for {me.Symbol}.");

        if (me.FactorScores is null)
            return BloodScreening.Reject($"FactorScores is null for {me.Symbol}.");

        if (me.Composite is null)
            return BloodScreening.Reject($"Composite is null for {me.Symbol}.");

        // Validate factor scores are in bounds [-1, 1]
        var factorCheck = ScreenFactorScores(me.FactorScores, me.Symbol);
        if (!factorCheck.IsHealthy)
            return factorCheck;

        // Validate composite probabilities are sane
        var compositeCheck = ScreenComposite(me.Composite, me.Symbol);
        if (!compositeCheck.IsHealthy)
            return compositeCheck;

        // Validate confidence is in [0, 1]
        if (me.Confidence < 0 || me.Confidence > 1)
            return BloodScreening.Reject($"Confidence out of range ({me.Confidence:F4}) for {me.Symbol}.");

        // If temporal envelope is present, validate it
        if (me.Temporal is not null)
        {
            var temporalCheck = ScreenTemporal(me.Temporal, me.Symbol);
            if (!temporalCheck.IsHealthy)
                return temporalCheck;
        }

        return BloodScreening.Healthy;
    }

    // ── PredictionEvent screening ──

    public static BloodScreening Screen(PredictionEvent pe)
    {
        if (pe is null)
            return BloodScreening.Reject("Event is null.");

        if (string.IsNullOrWhiteSpace(pe.Symbol))
            return BloodScreening.Reject("Symbol is empty.");

        if (!SymbolValidator.IsValid(pe.Symbol))
            return BloodScreening.Reject($"Symbol '{pe.Symbol}' failed validation.");

        if (pe.Probabilities is null)
            return BloodScreening.Reject($"Probabilities is null for {pe.Symbol}.");

        // Validate probability distribution sums approximately to 1.0
        var probSum = pe.Probabilities.Bullish + pe.Probabilities.Neutral + pe.Probabilities.Bearish;
        if (probSum < 0.90 || probSum > 1.10)
            return BloodScreening.Reject(
                $"Probability sum out of range ({probSum:F4}) for {pe.Symbol}. " +
                $"Bull={pe.Probabilities.Bullish:F3} Neut={pe.Probabilities.Neutral:F3} Bear={pe.Probabilities.Bearish:F3}");

        // Confidence must be [0, 1]
        if (pe.Confidence < 0 || pe.Confidence > 1)
            return BloodScreening.Reject($"Confidence out of range ({pe.Confidence:F4}) for {pe.Symbol}.");

        // ActionTendency must be [-1, 1]
        if (pe.ActionTendency < -1 || pe.ActionTendency > 1)
            return BloodScreening.Reject($"ActionTendency out of range ({pe.ActionTendency:F4}) for {pe.Symbol}.");

        // PredictedClass must be a known value
        if (pe.PredictedClass is not "bullish" and not "neutral" and not "bearish")
            return BloodScreening.Reject($"Unknown PredictedClass '{pe.PredictedClass}' for {pe.Symbol}.");

        // NaN/Infinity checks on all numeric fields
        if (double.IsNaN(pe.Confidence) || double.IsInfinity(pe.Confidence))
            return BloodScreening.Reject($"Confidence is NaN/Infinity for {pe.Symbol}.");

        if (double.IsNaN(pe.ActionTendency) || double.IsInfinity(pe.ActionTendency))
            return BloodScreening.Reject($"ActionTendency is NaN/Infinity for {pe.Symbol}.");

        // Temporal security invariant: if temporal failed, training must be off
        if (!pe.TemporalSecurityPassed && pe.EligibleForTraining)
            return BloodScreening.Reject(
                $"Temporal security violation: training eligible despite failed temporal check for {pe.Symbol}.");

        return BloodScreening.Healthy;
    }

    // ── Sub-screenings ──

    private static BloodScreening ScreenFactorScores(MetabolicFactorScores fs, string symbol)
    {
        var factors = new[]
        {
            ("Trend", fs.Trend),
            ("Momentum", fs.Momentum),
            ("Volatility", fs.Volatility),
            ("Participation", fs.Participation),
        };

        foreach (var (name, factor) in factors)
        {
            if (factor is null)
                return BloodScreening.Reject($"{name} factor is null for {symbol}.");

            if (double.IsNaN(factor.Score) || double.IsInfinity(factor.Score))
                return BloodScreening.Reject($"{name} factor score is NaN/Infinity for {symbol}.");

            if (factor.Score < -1 || factor.Score > 1)
                return BloodScreening.Reject($"{name} factor score out of [-1,1] range ({factor.Score:F4}) for {symbol}.");
        }

        return BloodScreening.Healthy;
    }

    private static BloodScreening ScreenComposite(MetabolicComposite comp, string symbol)
    {
        var probs = new[]
        {
            ("BullishProbability", comp.BullishProbability),
            ("BearishProbability", comp.BearishProbability),
            ("NeutralProbability", comp.NeutralProbability),
            ("Confidence", comp.Confidence),
        };

        foreach (var (name, val) in probs)
        {
            if (double.IsNaN(val) || double.IsInfinity(val))
                return BloodScreening.Reject($"Composite {name} is NaN/Infinity for {symbol}.");

            if (val < 0 || val > 1)
                return BloodScreening.Reject($"Composite {name} out of [0,1] range ({val:F4}) for {symbol}.");
        }

        var probSum = comp.BullishProbability + comp.BearishProbability + comp.NeutralProbability;
        if (probSum < 0.90 || probSum > 1.10)
            return BloodScreening.Reject($"Composite probability sum out of range ({probSum:F4}) for {symbol}.");

        return BloodScreening.Healthy;
    }

    private static BloodScreening ScreenTemporal(MetabolicTemporalEnvelope te, string symbol)
    {
        if (string.IsNullOrWhiteSpace(te.ObservationCutoffUtc))
            return BloodScreening.Reject($"Temporal ObservationCutoffUtc is empty for {symbol}.");

        if (string.IsNullOrWhiteSpace(te.TemporalPolicyVersion))
            return BloodScreening.Reject($"Temporal PolicyVersion is empty for {symbol}.");

        return BloodScreening.Healthy;
    }
}
