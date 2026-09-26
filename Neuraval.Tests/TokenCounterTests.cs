using Neuraval.ChatBot.Benchmarking;
using Xunit;

namespace Neuraval.Tests
{
    public class TokenCounterTests
    {
        [Fact]
        public void WhitespaceTokenCounter_CountsWordsSeparatedBySpaces()
        {
            var counter = new WhitespaceTokenCounter();

            Assert.Equal(3, counter.CountTokens("una  dos   tres"));
        }

        [Fact]
        public void WhitespaceTokenCounter_ReturnsZero_ForEmptyOrWhitespaceText()
        {
            var counter = new WhitespaceTokenCounter();

            Assert.Equal(0, counter.CountTokens(""));
            Assert.Equal(0, counter.CountTokens("   "));
        }

        [Fact]
        public void TokenizerTokenCounter_DelegatesToTokenizerEncode_WithoutSpecialTokens()
        {
            var counter = new TokenizerTokenCounter(new FakeTokenizer());

            Assert.Equal(2, counter.CountTokens("hola mundo"));
        }

        [Fact]
        public void TokenizerTokenCounter_ReturnsZero_ForEmptyText()
        {
            var counter = new TokenizerTokenCounter(new FakeTokenizer());

            Assert.Equal(0, counter.CountTokens(""));
        }
    }
}
