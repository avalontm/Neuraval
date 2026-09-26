using System;
using System.Linq;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization.SafeTensors;
using Neuraval.Core.Serialization.WeightLoading;
using Xunit;

namespace Neuraval.Tests
{
    public class TensorNameMapperValidatorTests
    {
        private static TransformerConfig BuildConfig(bool tieWordEmbeddings = false)
        {
            return new TransformerConfig
            {
                VocabSize = 32,
                HiddenSize = 8,
                NumHiddenLayers = 1,
                NumAttentionHeads = 2,
                NumKeyValueHeads = 1,
                IntermediateSize = 16,
                MaxPositionEmbeddings = 32,
                TieWordEmbeddings = tieWordEmbeddings
            };
        }

        [Fact]
        public void Validate_ExactMatch_IsComplete()
        {
            var config = BuildConfig();
            var names = TensorNameMapper.AllExpectedNames(config);

            var report = TensorNameMapperValidator.Validate(config, names);

            Assert.True(report.IsComplete);
            Assert.Empty(report.MissingNames);
            Assert.Empty(report.UnexpectedNames);
        }

        [Fact]
        public void Validate_MissingTensor_IsReported()
        {
            var config = BuildConfig();
            var names = TensorNameMapper.AllExpectedNames(config).Where(name => name != TensorNameMapper.FinalNormName);

            var report = TensorNameMapperValidator.Validate(config, names);

            Assert.False(report.IsComplete);
            Assert.Contains(TensorNameMapper.FinalNormName, report.MissingNames);
        }

        [Fact]
        public void Validate_UnexpectedTensor_IsReported()
        {
            var config = BuildConfig();
            var names = TensorNameMapper.AllExpectedNames(config).Append("model.layers.0.extra_tensor.weight");

            var report = TensorNameMapperValidator.Validate(config, names);

            Assert.True(report.IsComplete);
            Assert.Contains("model.layers.0.extra_tensor.weight", report.UnexpectedNames);
        }

        [Fact]
        public void Validate_TiedEmbeddings_DoesNotRequireLmHead()
        {
            var config = BuildConfig(tieWordEmbeddings: true);
            var names = TensorNameMapper.AllExpectedNames(config);

            var report = TensorNameMapperValidator.Validate(config, names);

            Assert.True(report.IsComplete);
            Assert.DoesNotContain(TensorNameMapper.LmHeadName, TensorNameMapper.AllExpectedNames(config));
        }

        [Fact]
        public void Validate_AgainstSafeTensorsFile_UsesTensorNames()
        {
            var config = BuildConfig();
            var entries = TensorNameMapper.AllExpectedNames(config)
                .Select(name => new SafeTensorsEntry(name, SafeTensorsDType.F32, new[] { 1 }, new float[] { 1f }))
                .ToArray();
            var file = new SafeTensorsFile(entries, new System.Collections.Generic.Dictionary<string, string>());

            var report = TensorNameMapperValidator.Validate(config, file);

            Assert.True(report.IsComplete);
        }

        [Fact]
        public void Validate_NullConfig_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => TensorNameMapperValidator.Validate(null!, Enumerable.Empty<string>()));
        }

        [Fact]
        public void Validate_NullAvailableNames_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => TensorNameMapperValidator.Validate(BuildConfig(), (System.Collections.Generic.IEnumerable<string>)null!));
        }

        [Fact]
        public void Validate_NullSafeTensorsFile_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => TensorNameMapperValidator.Validate(BuildConfig(), (SafeTensorsFile)null!));
        }
    }
}
