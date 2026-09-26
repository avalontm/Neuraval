using System;
using Neuraval.Core.Serialization.ModelExport;
using Xunit;

namespace Neuraval.Tests
{
    public class GenerationConfigJsonConverterTests
    {
        [Fact]
        public void ToJson_UsesSnakeCasePropertyNames()
        {
            var json = GenerationConfigJsonConverter.ToJson(GenerationConfigOptions.Default());

            Assert.Contains("\"temperature\"", json);
            Assert.Contains("\"top_p\"", json);
            Assert.Contains("\"top_k\"", json);
            Assert.Contains("\"max_new_tokens\"", json);
        }

        [Fact]
        public void FromJson_ToJson_RoundTrips()
        {
            var original = GenerationConfigOptions.Create(0.8f, 0.92f, 64, 128);

            var json = GenerationConfigJsonConverter.ToJson(original);
            var restored = GenerationConfigJsonConverter.FromJson(json);

            Assert.Equal(original.Temperature, restored.Temperature);
            Assert.Equal(original.TopP, restored.TopP);
            Assert.Equal(original.TopK, restored.TopK);
            Assert.Equal(original.MaxNewTokens, restored.MaxNewTokens);
        }

        [Fact]
        public void ToJson_NullOptions_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => GenerationConfigJsonConverter.ToJson(null!));
        }

        [Fact]
        public void FromJson_NullJson_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => GenerationConfigJsonConverter.FromJson(null!));
        }
    }
}
