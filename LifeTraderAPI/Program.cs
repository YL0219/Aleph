using Microsoft.EntityFrameworkCore;
using LifeTrader_AI.Data;
using LifeTrader_AI.Services;
using LifeTrader_AI.Services.Ingestion;
using LifeTrader_AI.Infrastructure;
using LifeTrader_AI.Infrastructure.Python;
using LifeTrader_AI.Infrastructure.Mcp;
using Serilog;

// --- 0. ENTERPRISE LOGGER (Serilog) ---
Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Debug()
    .MinimumLevel.Override("Microsoft", Serilog.Events.LogEventLevel.Warning)
    .MinimumLevel.Override("Microsoft.EntityFrameworkCore.Database.Command", Serilog.Events.LogEventLevel.Warning)
    .MinimumLevel.Override("System", Serilog.Events.LogEventLevel.Warning)
    .WriteTo.Console(restrictedToMinimumLevel: Serilog.Events.LogEventLevel.Information)
    .WriteTo.File("logs/deepblue_log_.txt",
        rollingInterval: RollingInterval.Day,
        retainedFileCountLimit: 7,
        restrictedToMinimumLevel: Serilog.Events.LogEventLevel.Debug)
    .CreateLogger();

var builder = WebApplication.CreateBuilder(args);

builder.Logging.ClearProviders();
builder.Host.UseSerilog();

// 1. CONTROLLERS
builder.Services.AddControllers();

// 1.5 DATABASE (EF Core + SQLite, Scoped per-request)
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlite("Data Source=deepblue.db"));

// 1.6 SERVICE LAYER (Scoped — shares DbContext lifetime)
builder.Services.AddScoped<TradingService>();

// 1.7 MEMORY CACHE (10s TTL, reduces redundant API calls)
builder.Services.AddMemoryCache();

// 1.8 PYTHON PATH RESOLVER (Singleton — resolves Python:ExePath once at startup)
// If the venv is missing, PythonPathResolver logs a warning but does NOT crash.
builder.Services.AddSingleton<PythonPathResolver>();

// 1.9 PYTHON DISPATCHER (Single gateway — the ONLY class that spawns Python processes)
// Encapsulates SemaphoreSlim(3) gating internally. Routes all calls through python_router.py.
builder.Services.AddSingleton<PythonDispatcherService>();

// 1.10 MARKET DATA INGESTION (background 5-min OHLCV cycle)
builder.Services.AddHostedService<MarketIngestionOrchestrator>();


// 1.11 MCP SERVER (Model Context Protocol — HTTP/SSE transport)
// Exposes market tools via MCP. Tools discovered from [McpServerToolType] classes in this assembly.
// Endpoint: /mcp (mapped below via app.MapMcp())
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

// 1.12 MCP ↔ OPENAI BRIDGE (translates MCP tool schemas for the OpenAI agent loop)
// McpMarketTools: singleton — deps are all singletons, no mutable state.
// McpToolSchemaAdapter: reflects on assembly to build OpenAI function schemas, caches result.
// McpToolInvoker: routes OpenAI tool_call dispatch to MCP tool methods.
builder.Services.AddSingleton<McpMarketTools>();
builder.Services.AddSingleton<McpExecutionTools>();
builder.Services.AddSingleton<McpToolSchemaAdapter>();
builder.Services.AddSingleton<McpToolInvoker>();

// 2. CORS (Unity frontend)
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

// 3. MIDDLEWARE PIPELINE
app.UseStaticFiles();
app.UseCors("AllowUnity");
app.MapControllers();
app.MapMcp();   // MCP SSE/HTTP endpoint at /mcp

app.Lifetime.ApplicationStopped.Register(Log.CloseAndFlush);

app.Run();
