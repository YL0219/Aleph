namespace Aleph;

/// <summary>
/// Lightweight contract for any biological organ in the Aleph organism.
/// Organs are bloodstream consumers that subscribe to the AlephBus,
/// process specific event types, and may publish refined events back.
///
/// This is NOT a workflow engine or generic pipeline abstraction.
/// It standardizes identity and subscription intent so the organism
/// can introspect its own organs at runtime.
/// </summary>
public interface IAlephOrgan
{
    /// <summary>Human-readable organ identity (e.g. "Liver", "Kidneys").</summary>
    string OrganName { get; }

    /// <summary>
    /// Declared event type names this organ is interested in.
    /// Used for documentation, introspection, and optional routing.
    /// The organ is responsible for its own AlephBus subscription filtering.
    /// </summary>
    IReadOnlyList<string> EventInterests { get; }

    /// <summary>Whether the organ is currently active and processing events.</summary>
    bool IsActive { get; }
}
