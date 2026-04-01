// CONTRACT / INVARIANTS
// - Valid symbols: 1-15 characters, uppercase A-Z, digits 0-9, dots, hyphens, equals.
//   Covers standard tickers (AAPL), class shares (BRK.B), hyphenated (BF-B),
//   and futures notation (SI=F).
// - Normalize() always returns UPPERCASE.
// - Must be used everywhere symbols enter Python args, external API calls, or cache keys.
// - Stateless, thread-safe, no allocations beyond the normalized string.

using System.Text.RegularExpressions;

namespace Aleph
{
    /// <summary>
    /// Centralized symbol validation and normalization.
    /// Prevents injection and ensures consistent format across all subsystems.
    /// </summary>
    public static class SymbolValidator
    {
        private static readonly Regex ValidPattern =
            new(@"^[A-Z0-9.\-=]{1,15}$", RegexOptions.Compiled);

        /// <summary>Checks if the symbol is valid (after uppercasing).</summary>
        public static bool IsValid(string? symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol)) return false;
            return ValidPattern.IsMatch(symbol.ToUpperInvariant());
        }

        /// <summary>Returns the symbol in uppercase. Does NOT validate — call IsValid first.</summary>
        public static string Normalize(string symbol) => symbol.ToUpperInvariant();

        /// <summary>
        /// Validates and normalizes in one call. Returns false if invalid.
        /// Preferred for input boundaries (controllers, tool argument parsing).
        /// </summary>
        public static bool TryNormalize(string? symbol, out string normalized)
        {
            normalized = "";
            if (string.IsNullOrWhiteSpace(symbol)) return false;
            normalized = symbol.ToUpperInvariant();
            return ValidPattern.IsMatch(normalized);
        }
    }
}
