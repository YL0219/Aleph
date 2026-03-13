using System.Threading.Channels;

namespace Aleph;

/// <summary>
/// Internal in-memory event bus with fan-out broadcast semantics.
/// One published AlephEvent is delivered to all active subscribers.
/// Channel-backed, singleton-safe.
/// </summary>
public interface IAlephBus
{
    /// <summary>Publish a blood cell to all subscribers. Non-blocking fire-and-forget semantics.</summary>
    ValueTask PublishAsync(AlephEvent evt, CancellationToken ct = default);

    /// <summary>
    /// Subscribe to the bus. Returns a ChannelReader that receives all published events
    /// (optionally filtered). Each subscriber gets its own bounded channel.
    /// </summary>
    ChannelReader<AlephEvent> Subscribe(string subscriberName, Func<AlephEvent, bool>? filter = null);
}
