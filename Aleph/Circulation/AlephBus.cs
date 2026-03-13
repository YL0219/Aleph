using System.Threading.Channels;

namespace Aleph;

/// <summary>
/// Singleton in-memory fan-out event bus backed by System.Threading.Channels.
/// Each subscriber gets its own bounded channel. Publish broadcasts to all subscribers.
/// If a subscriber's channel is full, the event is dropped for that subscriber (back-pressure).
/// </summary>
public sealed class AlephBus : IAlephBus
{
    private const int DefaultSubscriberCapacity = 1024;

    private readonly object _lock = new();
    private readonly List<Subscription> _subscriptions = new();
    private readonly ILogger<AlephBus> _logger;

    public AlephBus(ILogger<AlephBus> logger)
    {
        _logger = logger;
    }

    public ValueTask PublishAsync(AlephEvent evt, CancellationToken ct = default)
    {
        if (evt is null) return ValueTask.CompletedTask;

        List<Subscription> snapshot;
        lock (_lock)
        {
            snapshot = new List<Subscription>(_subscriptions);
        }

        foreach (var sub in snapshot)
        {
            // Apply subscriber filter
            if (sub.Filter is not null && !sub.Filter(evt))
                continue;

            // TryWrite is non-blocking — if the channel is full, drop silently
            if (!sub.Writer.TryWrite(evt))
            {
                _logger.LogWarning(
                    "[AlephBus] Dropped event {Kind} for subscriber '{Name}' (channel full).",
                    evt.Kind, sub.Name);
            }
        }

        return ValueTask.CompletedTask;
    }

    public ChannelReader<AlephEvent> Subscribe(string subscriberName, Func<AlephEvent, bool>? filter = null)
    {
        var channel = Channel.CreateBounded<AlephEvent>(new BoundedChannelOptions(DefaultSubscriberCapacity)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
            SingleWriter = false
        });

        var subscription = new Subscription(subscriberName, channel.Writer, filter);

        lock (_lock)
        {
            _subscriptions.Add(subscription);
        }

        _logger.LogInformation("[AlephBus] Subscriber '{Name}' registered.", subscriberName);

        return channel.Reader;
    }

    private sealed record Subscription(
        string Name,
        ChannelWriter<AlephEvent> Writer,
        Func<AlephEvent, bool>? Filter);
}
