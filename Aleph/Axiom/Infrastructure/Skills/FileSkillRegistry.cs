// CONTRACT / INVARIANTS
// - Reads .md files ONLY from <ContentRoot>/Brain/Skills (no subdirectories, no path traversal).
// - YAML frontmatter parsed between --- delimiters; unknown keys captured in Extras with soft warning.
// - SHA256 hash computed on raw file bytes.
// - File size hard cap: 256 KiB. Warning threshold: 128 KiB.
// - Duplicate skill_name (case-insensitive): hard error — both files rejected.
// - Snapshot swap is atomic via Interlocked.Exchange; current snapshot always valid (empty on first init).
// - LoadAsync is non-fatal: on catastrophic failure, logs and retains previous snapshot.

using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using YamlDotNet.Serialization;

namespace Aleph
{
    public sealed class FileSkillRegistry : ISkillRegistry
    {
        private const long WarnSizeBytes = 128 * 1024;
        private const long MaxSizeBytes = 256 * 1024;

        private static readonly Regex SkillNamePattern =
            new(@"^[a-z][a-z0-9_]{0,63}$", RegexOptions.Compiled);

        private static readonly HashSet<string> KnownYamlKeys =
            new(StringComparer.OrdinalIgnoreCase)
            {
                "skill_name", "display_name", "version", "description",
                "tags", "required_tools", "deprecated"
            };

        private readonly string _skillsRoot;
        private readonly ILogger<FileSkillRegistry> _logger;
        private readonly IDeserializer _yaml;

        private SkillRegistrySnapshot _snapshot = SkillRegistrySnapshot.Empty;
        public SkillRegistrySnapshot Snapshot => Volatile.Read(ref _snapshot);

        public FileSkillRegistry(
            IWebHostEnvironment env,
            ILogger<FileSkillRegistry> logger)
        {
            _skillsRoot = Path.GetFullPath(Path.Combine(env.ContentRootPath, "Arbiter", "Skills"));
            _logger = logger;
            _yaml = new DeserializerBuilder().Build();
        }

        public async Task LoadAsync(CancellationToken ct = default)
        {
            _logger.LogInformation("[SkillRegistry] Loading skills from {Root}", _skillsRoot);

            if (!Directory.Exists(_skillsRoot))
            {
                _logger.LogWarning("[SkillRegistry] Skills directory not found: {Root}", _skillsRoot);
                Interlocked.Exchange(ref _snapshot, SkillRegistrySnapshot.Empty);
                return;
            }

            var files = Directory.GetFiles(_skillsRoot, "*.md", SearchOption.TopDirectoryOnly);
            var candidates = new List<SkillPlaybook>();
            int errorCount = 0;

            foreach (var filePath in files)
            {
                ct.ThrowIfCancellationRequested();

                var fullPath = Path.GetFullPath(filePath);
                if (!fullPath.StartsWith(_skillsRoot, StringComparison.OrdinalIgnoreCase))
                {
                    _logger.LogWarning("[SkillRegistry] Path traversal blocked: {Path}", filePath);
                    continue;
                }

                try
                {
                    var playbook = await ParseFileAsync(fullPath, ct);
                    if (playbook != null)
                        candidates.Add(playbook);
                    else
                        errorCount++;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[SkillRegistry] Failed to parse: {Path}", fullPath);
                    errorCount++;
                }
            }

            // Duplicate detection — case-insensitive; both rejected on conflict
            var valid = new List<SkillPlaybook>();
            int duplicateCount = 0;

            foreach (var group in candidates.GroupBy(
                         p => p.Metadata.SkillName, StringComparer.OrdinalIgnoreCase))
            {
                var items = group.ToList();
                if (items.Count > 1)
                {
                    _logger.LogError(
                        "[SkillRegistry] Duplicate skill_name '{SkillName}' in {Count} files — both rejected.",
                        group.Key, items.Count);
                    duplicateCount += items.Count;
                }
                else
                {
                    valid.Add(items[0]);
                }
            }

            // Atomic snapshot swap — only after full candidate build succeeds
            var newSnapshot = new SkillRegistrySnapshot(valid);
            Interlocked.Exchange(ref _snapshot, newSnapshot);

            _logger.LogInformation(
                "[SkillRegistry] Loaded {Count} skills ({Errors} parse errors, {Dupes} duplicates rejected).",
                valid.Count, errorCount, duplicateCount);
        }

        private async Task<SkillPlaybook?> ParseFileAsync(string filePath, CancellationToken ct)
        {
            var fileBytes = await File.ReadAllBytesAsync(filePath, ct);
            long fileSize = fileBytes.Length;

            // --- Size checks ---
            if (fileSize > MaxSizeBytes)
            {
                _logger.LogError(
                    "[SkillRegistry] File exceeds 256 KiB hard cap ({Size} bytes): {Path}",
                    fileSize, filePath);
                return null;
            }

            if (fileSize > WarnSizeBytes)
            {
                _logger.LogWarning(
                    "[SkillRegistry] File exceeds 128 KiB warning threshold ({Size} bytes): {Path}",
                    fileSize, filePath);
            }

            // --- Content hash ---
            var hashBytes = SHA256.HashData(fileBytes);
            var contentHash = $"sha256:{Convert.ToHexString(hashBytes).ToLowerInvariant()}";

            // --- Extract frontmatter + body ---
            var content = Encoding.UTF8.GetString(fileBytes);
            if (!TryExtractFrontmatter(content, out var yamlText, out var markdownBody))
            {
                _logger.LogWarning("[SkillRegistry] No YAML frontmatter in: {Path}", filePath);
                return null;
            }

            // --- Parse YAML ---
            Dictionary<string, object>? yamlDict;
            try
            {
                yamlDict = _yaml.Deserialize<Dictionary<string, object>>(yamlText);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[SkillRegistry] Invalid YAML in: {Path}", filePath);
                return null;
            }

            if (yamlDict == null || yamlDict.Count == 0)
            {
                _logger.LogWarning("[SkillRegistry] Empty YAML frontmatter in: {Path}", filePath);
                return null;
            }

            // --- Extract required fields ---
            var skillName = GetString(yamlDict, "skill_name");
            var displayName = GetString(yamlDict, "display_name");
            var version = GetString(yamlDict, "version");
            var description = GetString(yamlDict, "description");

            if (string.IsNullOrWhiteSpace(skillName))
            {
                _logger.LogError("[SkillRegistry] Missing 'skill_name' in: {Path}", filePath);
                return null;
            }

            if (!SkillNamePattern.IsMatch(skillName))
            {
                _logger.LogError("[SkillRegistry] Invalid skill_name '{Name}' in: {Path}", skillName, filePath);
                return null;
            }

            if (string.IsNullOrWhiteSpace(displayName))
            {
                _logger.LogError("[SkillRegistry] Missing 'display_name' in: {Path}", filePath);
                return null;
            }

            if (string.IsNullOrWhiteSpace(version))
            {
                _logger.LogError("[SkillRegistry] Missing 'version' in: {Path}", filePath);
                return null;
            }

            if (string.IsNullOrWhiteSpace(description))
            {
                _logger.LogError("[SkillRegistry] Missing 'description' in: {Path}", filePath);
                return null;
            }

            // --- Extract optional fields ---
            var tags = GetStringList(yamlDict, "tags");
            var requiredTools = GetStringList(yamlDict, "required_tools");
            var deprecated = GetBool(yamlDict, "deprecated");

            // --- Collect unknown keys → extras + soft warning ---
            var extras = new Dictionary<string, object>();
            foreach (var kv in yamlDict)
            {
                if (!KnownYamlKeys.Contains(kv.Key))
                {
                    extras[kv.Key] = kv.Value;
                    _logger.LogWarning(
                        "[SkillRegistry] Unknown YAML key '{Key}' in: {Path} — captured in extras.",
                        kv.Key, filePath);
                }
            }

            var metadata = new SkillMetadata
            {
                SkillName = skillName,
                DisplayName = displayName,
                Version = version,
                Description = description,
                Tags = tags,
                RequiredTools = requiredTools,
                Deprecated = deprecated,
                Extras = extras
            };

            return new SkillPlaybook
            {
                Metadata = metadata,
                MarkdownBody = markdownBody.Trim(),
                ContentHash = contentHash,
                FileSizeBytes = fileSize
            };
        }

        // ─── Frontmatter Extraction ───────────────────────────────────────

        private static bool TryExtractFrontmatter(
            string content,
            out string yamlText,
            out string markdownBody)
        {
            yamlText = "";
            markdownBody = "";

            var trimmed = content.TrimStart();
            if (!trimmed.StartsWith("---"))
                return false;

            // Skip past the opening "---" line
            int firstNewline = trimmed.IndexOf('\n');
            if (firstNewline < 0) return false;
            var afterOpening = trimmed[(firstNewline + 1)..];

            // Find closing "---"
            int closingIndex = afterOpening.IndexOf("\n---", StringComparison.Ordinal);
            if (closingIndex < 0) return false;

            yamlText = afterOpening[..closingIndex].Trim();

            // Everything after the closing "---" line is the markdown body
            var afterClosing = afterOpening[(closingIndex + 4)..]; // skip "\n---"
            int bodyStart = afterClosing.IndexOf('\n');
            markdownBody = bodyStart >= 0 ? afterClosing[(bodyStart + 1)..] : "";

            return !string.IsNullOrWhiteSpace(yamlText);
        }

        // ─── YAML Value Helpers ───────────────────────────────────────────

        private static string GetString(Dictionary<string, object> dict, string key)
        {
            if (!dict.TryGetValue(key, out var val) || val == null) return "";
            return val.ToString() ?? "";
        }

        private static List<string> GetStringList(Dictionary<string, object> dict, string key)
        {
            if (!dict.TryGetValue(key, out var val) || val == null)
                return new();

            if (val is string)
                return new();

            if (val is not System.Collections.IEnumerable enumerable)
                return new();

            var result = new List<string>();
            foreach (var item in enumerable)
            {
                var s = item?.ToString();
                if (!string.IsNullOrEmpty(s))
                    result.Add(s);
            }
            return result;
        }

        private static bool GetBool(Dictionary<string, object> dict, string key)
        {
            if (!dict.TryGetValue(key, out var val) || val == null) return false;
            if (val is bool b) return b;
            if (val is string s) return bool.TryParse(s, out var parsed) && parsed;
            return false;
        }
    }
}
