using Microsoft.EntityFrameworkCore;

namespace Aleph
{
    /// <summary>
    /// The central EF Core database context for Deep Blue.
    /// Manages the Positions, Trades, and ChatMessages tables in a SQLite database.
    /// Registered as Scoped in Program.cs — each HTTP request gets its own instance (thread-safe).
    /// </summary>
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
        {
        }

        /// <summary>
        /// Stock positions held in the portfolio. Replaces portfolio.json.
        /// </summary>
        public DbSet<Position> Positions { get; set; } = null!;

        /// <summary>
        /// Individual trade executions with idempotency keys.
        /// </summary>
        public DbSet<Trade> Trades { get; set; } = null!;

        /// <summary>
        /// OpenAI conversation history. Replaces trade_log.txt.
        /// </summary>
        public DbSet<ChatMessage> ChatMessages { get; set; } = null!;

        /// <summary>
        /// Observability log for every tool invocation in the AI agent loop.
        /// </summary>
        public DbSet<ToolRun> ToolRuns { get; set; } = null!;

        /// <summary>
        /// User watchlist symbols — active items are included in market data ingestion.
        /// </summary>
        public DbSet<WatchlistItem> WatchlistItems { get; set; } = null!;

        /// <summary>
        /// Metadata for ingested OHLCV Parquet files (one row per symbol+interval).
        /// </summary>
        public DbSet<MarketDataAsset> MarketDataAssets { get; set; } = null!;

        /// <summary>
        /// Persisted autonomic and heartbeat events (written by the Kidneys).
        /// </summary>
        public DbSet<AutonomicEvent> AutonomicEvents { get; set; } = null!;

        /// <summary>
        /// Auto-increments RowVersion on any modified Position entity before saving.
        /// This ensures the concurrency token is always updated, even if the service
        /// layer forgets to do it manually.
        /// </summary>
        public override async Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
        {
            foreach (var entry in ChangeTracker.Entries<Position>())
            {
                if (entry.State == EntityState.Modified)
                {
                    entry.Entity.UpdatedAtUtc = DateTime.UtcNow;
                    entry.Entity.RowVersion++;
                }
            }

            return await base.SaveChangesAsync(cancellationToken);
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // === Position Configuration ===
            modelBuilder.Entity<Position>(entity =>
            {
                entity.ToTable("Positions");

                // Composite unique: one row per Symbol+Currency pair
                entity.HasIndex(p => new { p.Symbol, p.Currency })
                      .IsUnique()
                      .HasDatabaseName("IX_Positions_Symbol_Currency");

                entity.Property(p => p.Quantity)
                      .HasDefaultValue(0m);

                entity.Property(p => p.IsOpen)
                      .HasDefaultValue(true);

                // Optimistic concurrency token for SQLite
                // EF Core adds WHERE RowVersion = @old to UPDATE statements
                entity.Property(p => p.RowVersion)
                      .IsConcurrencyToken();

                entity.Property(p => p.CreatedAtUtc)
                      .HasDefaultValueSql("datetime('now')");

                entity.Property(p => p.UpdatedAtUtc)
                      .HasDefaultValueSql("datetime('now')");
            });

            // === Trade Configuration ===
            modelBuilder.Entity<Trade>(entity =>
            {
                entity.ToTable("Trades", t =>
                {
                    // Only allow valid trade sides
                    t.HasCheckConstraint(
                        "CK_Trades_Side",
                        "[Side] IN ('BUY', 'SELL')"
                    );

                    // Only allow valid trade statuses
                    t.HasCheckConstraint(
                        "CK_Trades_Status",
                        "[Status] IN ('FILLED', 'PENDING', 'REJECTED')"
                    );
                });

                // Idempotency: one trade per ClientRequestId
                entity.HasIndex(t => t.ClientRequestId)
                      .IsUnique()
                      .HasDatabaseName("IX_Trades_ClientRequestId");

                // Index for querying trades by symbol
                entity.HasIndex(t => t.Symbol)
                      .HasDatabaseName("IX_Trades_Symbol");

                // Index for querying by execution time
                entity.HasIndex(t => t.ExecutedAtUtc)
                      .HasDatabaseName("IX_Trades_ExecutedAtUtc");

                // FK to Position (optional)
                entity.HasOne(t => t.Position)
                      .WithMany(p => p.Trades)
                      .HasForeignKey(t => t.PositionId)
                      .OnDelete(DeleteBehavior.SetNull);

                entity.Property(t => t.ExecutedAtUtc)
                      .HasDefaultValueSql("datetime('now')");
            });

            // === ChatMessage Configuration ===
            modelBuilder.Entity<ChatMessage>(entity =>
            {
                entity.ToTable("ChatMessages", t =>
                {
                    // Only allow valid OpenAI roles
                    t.HasCheckConstraint(
                        "CK_ChatMessages_Role",
                        "[Role] IN ('system', 'user', 'assistant', 'tool')"
                    );
                });

                // Index for filtering by conversation thread
                entity.HasIndex(c => c.ThreadId)
                      .HasDatabaseName("IX_ChatMessages_ThreadId");

                // Index for efficient time-based pruning queries
                entity.HasIndex(c => c.CreatedAtUtc)
                      .HasDatabaseName("IX_ChatMessages_CreatedAtUtc");

                // FK to Position (optional)
                entity.HasOne(c => c.Position)
                      .WithMany(p => p.ChatMessages)
                      .HasForeignKey(c => c.PositionId)
                      .OnDelete(DeleteBehavior.SetNull);

                entity.Property(c => c.CreatedAtUtc)
                      .HasDefaultValueSql("datetime('now')");
            });

            // === ToolRun Configuration ===
            modelBuilder.Entity<ToolRun>(entity =>
            {
                entity.ToTable("ToolRuns");

                entity.HasIndex(t => t.ThreadId)
                      .HasDatabaseName("IX_ToolRuns_ThreadId");

                entity.HasIndex(t => t.CreatedAtUtc)
                      .HasDatabaseName("IX_ToolRuns_CreatedAtUtc");

                entity.Property(t => t.CreatedAtUtc)
                      .HasDefaultValueSql("datetime('now')");
            });

            // === WatchlistItem Configuration ===
            modelBuilder.Entity<WatchlistItem>(entity =>
            {
                entity.ToTable("WatchlistItems");

                entity.HasIndex(w => w.Symbol)
                      .HasDatabaseName("IX_WatchlistItems_Symbol");

                entity.Property(w => w.AddedAtUtc)
                      .HasDefaultValueSql("datetime('now')");
            });

            // === MarketDataAsset Configuration ===
            modelBuilder.Entity<MarketDataAsset>(entity =>
            {
                entity.ToTable("MarketDataAssets");

                // Unique: one metadata row per symbol+interval pair
                entity.HasIndex(a => new { a.Symbol, a.Interval })
                      .IsUnique()
                      .HasDatabaseName("IX_MarketDataAssets_Symbol_Interval");

                entity.Property(a => a.ConsecutiveFailures)
                      .HasDefaultValue(0);

                entity.Property(a => a.RowsWritten)
                      .HasDefaultValue(0);

                entity.Property(a => a.UpdatedAtUtc)
                      .HasDefaultValueSql("datetime('now')");
            });

            // === AutonomicEvent Configuration (Kidneys) ===
            modelBuilder.Entity<AutonomicEvent>(entity =>
            {
                entity.ToTable("AutonomicEvents");

                entity.HasIndex(e => e.OccurredAtUtc)
                      .HasDatabaseName("IX_AutonomicEvents_OccurredAtUtc");

                entity.HasIndex(e => e.Kind)
                      .HasDatabaseName("IX_AutonomicEvents_Kind");

                entity.Property(e => e.OccurredAtUtc)
                      .HasDefaultValueSql("datetime('now')");
            });

        }
    }
}
