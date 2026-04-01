using Xunit;

namespace Aleph.Tests;

public class SymbolValidatorTests
{
    [Theory]
    [InlineData("SI=F")]
    [InlineData("GC=F")]
    [InlineData("CL=F")]
    [InlineData("AAPL")]
    [InlineData("BRK.B")]
    [InlineData("BF-B")]
    public void IsValid_AcceptsSupportedSymbolFormats(string symbol)
    {
        Assert.True(SymbolValidator.IsValid(symbol));
    }

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("SI/F")]
    [InlineData("SI F")]
    [InlineData("SI@F")]
    [InlineData("SYMBOL-IS-TOO-LONG")]
    public void IsValid_RejectsInvalidSymbolFormats(string symbol)
    {
        Assert.False(SymbolValidator.IsValid(symbol));
    }

    [Fact]
    public void TryNormalize_NormalizesFuturesSymbolToUppercase()
    {
        var ok = SymbolValidator.TryNormalize("si=f", out var normalized);

        Assert.True(ok);
        Assert.Equal("SI=F", normalized);
    }
}
