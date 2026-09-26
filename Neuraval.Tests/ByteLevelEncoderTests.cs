using Neuraval.Core.Tokenizers;
using Xunit;

namespace Neuraval.Tests
{
    public class ByteLevelEncoderTests
    {
        [Theory]
        [InlineData("Hello World")]
        [InlineData("hello world")]
        [InlineData("HELLO")]
        [InlineData("")]
        [InlineData("  leading and trailing  ")]
        [InlineData("línea con ñ y acentos áéíóú")]
        [InlineData("emoji 🚀 test")]
        [InlineData("tabs\tand\nnewlines")]
        public void EncodeDecode_RoundTrips_ExactOriginalText(string text)
        {
            var encoded = ByteLevelEncoder.Encode(text);
            var decoded = ByteLevelEncoder.Decode(encoded);

            Assert.Equal(text, decoded);
        }

        [Fact]
        public void Encode_PreservesCase_UpperAndLowerProduceDifferentOutput()
        {
            var lower = ByteLevelEncoder.Encode("hello");
            var upper = ByteLevelEncoder.Encode("HELLO");

            Assert.NotEqual(lower, upper);
        }

        [Fact]
        public void Alphabet_HasExactly256Symbols()
        {
            Assert.Equal(256, ByteLevelEncoder.Alphabet.Count);
        }

        [Fact]
        public void Alphabet_HasNoDuplicateSymbols()
        {
            var distinct = new HashSet<char>(ByteLevelEncoder.Alphabet);

            Assert.Equal(256, distinct.Count);
        }
    }
}
