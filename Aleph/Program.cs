using Aleph;
using Microsoft.EntityFrameworkCore;
using Serilog;

Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Debug()
    .MinimumLevel.Override("Microsoft", Serilog.Events.LogEventLevel.Warning)
    .MinimumLevel.Override("Microsoft.EntityFrameworkCore.Database.Command", Serilog.Events.LogEventLevel.Warning)
    .MinimumLevel.Override("System", Serilog.Events.LogEventLevel.Warning)
    .WriteTo.Console(restrictedToMinimumLevel: Serilog.Events.LogEventLevel.Information)
    .WriteTo.File(
        "logs/deepblue_log_.txt",
        rollingInterval: RollingInterval.Day,
        retainedFileCountLimit: 7,
        restrictedToMinimumLevel: Serilog.Events.LogEventLevel.Debug)
    .CreateLogger();

var builder = WebApplication.CreateBuilder(args);

builder.Logging.ClearProviders();
builder.Host.UseSerilog();

builder.Services.AddControllers();

var sqliteConnectionString =
    builder.Configuration.GetConnectionString("Aleph")
    ?? builder.Configuration.GetConnectionString("DefaultConnection")
    ?? "Data Source=deepblue.db";

// AddDbContextFactory automatically registers both the Singleton Factory 
// AND the Scoped DbContext for your legacy controllers.
builder.Services.AddDbContextFactory<AppDbContext>(options =>
    options.UseSqlite(sqliteConnectionString));

builder.Services.AddScoped<TradingService>();

builder.Services.AddMemoryCache();

builder.Services.AddSingleton<PythonPathResolver>();
builder.Services.AddSingleton<PythonDispatcherService>();

// AlephBus — singleton in-memory fan-out event bus (the Veins).
builder.Services.AddSingleton<AlephBus>();
builder.Services.AddSingleton<IAlephBus>(sp => sp.GetRequiredService<AlephBus>());

// Homeostasis singleton — serves both IHomeostasis (full read/write for Heartbeat)
// and IStressInjector (narrow write-only for external domains).
// Now receives IAlephBus to publish circulatory events.
builder.Services.AddSingleton<Homeostasis>();
builder.Services.AddSingleton<IHomeostasis>(sp => sp.GetRequiredService<Homeostasis>());
builder.Services.AddSingleton<IStressInjector>(sp => sp.GetRequiredService<Homeostasis>());

builder.Services.AddScoped<IMarketStressDetector, MarketStressDetector>();
builder.Services.AddScoped<IMarketIngestionCycle, MarketIngestionOrchestrator>();
builder.Services.AddHostedService<HeartbeatService>();

// Kidneys — background persistence consumer for autonomic/heartbeat events.
builder.Services.AddHostedService<AutonomicPersistenceService>();

builder.Services
    .AddMcpServer(options =>
    {
        options.ServerInfo = new()
        {
            Name = "DeepBlue",
            Version = "1.0.0"
        };
    })
    .WithHttpTransport()
    .WithToolsFromAssembly();

builder.Services.AddSingleton<ISkillRegistry, FileSkillRegistry>();

builder.Services.AddSingleton<McpMarketTools>();
builder.Services.AddSingleton<McpExecutionTools>();
builder.Services.AddSingleton<McpNewsTools>();
builder.Services.AddSingleton<McpSkillTools>();
builder.Services.AddSingleton<McpAetherTools>();
builder.Services.AddSingleton<McpToolSchemaAdapter>();
builder.Services.AddSingleton<McpToolInvoker>();

builder.Services.AddSingleton<IAxiom, Axiom>();
builder.Services.AddSingleton<IArbiter, Arbiter>();
builder.Services.AddSingleton<IAether, Aether>();

builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowUnity", policy =>
    {
        policy.AllowAnyOrigin()
            .AllowAnyMethod()
            .AllowAnyHeader();
    });
});

var app = builder.Build();

app.UseStaticFiles();
app.UseCors("AllowUnity");
app.MapControllers();
app.MapMcp();

app.Lifetime.ApplicationStopped.Register(Log.CloseAndFlush);

try
{
    var skillRegistry = app.Services.GetRequiredService<ISkillRegistry>();
    skillRegistry.LoadAsync().GetAwaiter().GetResult();
}
catch (Exception ex)
{
    Log.Error(ex, "[Startup] Skill registry load failed - continuing with empty snapshot.");
}

app.Run();
