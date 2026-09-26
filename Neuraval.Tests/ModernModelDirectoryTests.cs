using System;
using System.IO;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization.ModelExport;
using Neuraval.Core.Serialization.SafeTensors;
using Xunit;

namespace Neuraval.Tests
{
    public class ModernModelDirectoryTests
    {
        private static TransformerConfig BuildConfig()
        {
            return new TransformerConfig
            {
                Architecture = "neuraval-decoder",
                VocabSize = 128,
                HiddenSize = 16,
                NumHiddenLayers = 2,
                NumAttentionHeads = 4,
                NumKeyValueHeads = 2,
                IntermediateSize = 32,
                MaxPositionEmbeddings = 64
            };
        }

        private static SafeTensorsEntry[] BuildTensors()
        {
            return new[]
            {
                new SafeTensorsEntry("embed_tokens.weight", SafeTensorsDType.F32, new[] { 4, 2 }, new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }),
                new SafeTensorsEntry("layers.0.self_attn.q_proj.weight", SafeTensorsDType.F32, new[] { 2, 2 }, new float[] { 0.1f, 0.2f, 0.3f, 0.4f })
            };
        }

        private static string TempDirectory()
        {
            return Path.Combine(Path.GetTempPath(), $"navm_model_dir_{Guid.NewGuid():N}");
        }

        [Fact]
        public void Write_CreatesExpectedFiles()
        {
            var directory = TempDirectory();
            try
            {
                ModernModelDirectoryWriter.Write(directory, BuildConfig(), BuildTensors());

                Assert.True(File.Exists(Path.Combine(directory, "config.json")));
                Assert.True(File.Exists(Path.Combine(directory, "model.safetensors")));
                Assert.True(File.Exists(Path.Combine(directory, "generation_config.json")));
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
            }
        }

        [Fact]
        public void WriteRead_RoundTrip_PreservesConfigWeightsAndGenerationConfig()
        {
            var directory = TempDirectory();
            try
            {
                var config = BuildConfig();
                var tensors = BuildTensors();
                var generationConfig = GenerationConfigOptions.Create(0.5f, 0.8f, 20, 64);

                ModernModelDirectoryWriter.Write(directory, config, tensors, generationConfig);
                var loaded = ModernModelDirectoryReader.Read(directory);

                Assert.Equal(config.VocabSize, loaded.Config.VocabSize);
                Assert.Equal(config.HiddenSize, loaded.Config.HiddenSize);
                Assert.Equal(config.NumHiddenLayers, loaded.Config.NumHiddenLayers);

                Assert.Equal(2, loaded.Weights.Tensors.Count);
                Assert.Equal(tensors[0].Data, loaded.Weights.Find("embed_tokens.weight")!.Data);
                Assert.Equal(tensors[1].Data, loaded.Weights.Find("layers.0.self_attn.q_proj.weight")!.Data);

                Assert.Equal(0.5f, loaded.GenerationConfig.Temperature);
                Assert.Equal(0.8f, loaded.GenerationConfig.TopP);
                Assert.Equal(20, loaded.GenerationConfig.TopK);
                Assert.Equal(64, loaded.GenerationConfig.MaxNewTokens);
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
            }
        }

        [Fact]
        public void Write_WithoutGenerationConfig_UsesDefault()
        {
            var directory = TempDirectory();
            try
            {
                ModernModelDirectoryWriter.Write(directory, BuildConfig(), BuildTensors());
                var loaded = ModernModelDirectoryReader.Read(directory);

                Assert.Equal(GenerationConfigOptions.Default().Temperature, loaded.GenerationConfig.Temperature);
                Assert.Equal(GenerationConfigOptions.Default().MaxNewTokens, loaded.GenerationConfig.MaxNewTokens);
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
            }
        }

        [Fact]
        public void Write_InvalidConfig_Throws()
        {
            var directory = TempDirectory();
            var invalidConfig = new TransformerConfig
            {
                VocabSize = 0,
                HiddenSize = 16,
                NumHiddenLayers = 2,
                NumAttentionHeads = 4,
                NumKeyValueHeads = 2,
                IntermediateSize = 32,
                MaxPositionEmbeddings = 64
            };

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                ModernModelDirectoryWriter.Write(directory, invalidConfig, BuildTensors()));
        }

        [Fact]
        public void Read_MissingDirectory_Throws()
        {
            Assert.Throws<DirectoryNotFoundException>(() => ModernModelDirectoryReader.Read(TempDirectory()));
        }

        [Fact]
        public void Write_NullConfig_Throws()
        {
            var directory = TempDirectory();
            Assert.Throws<ArgumentNullException>(() => ModernModelDirectoryWriter.Write(directory, null!, BuildTensors()));
        }

        [Fact]
        public void Write_NullTensors_Throws()
        {
            var directory = TempDirectory();
            Assert.Throws<ArgumentNullException>(() => ModernModelDirectoryWriter.Write(directory, BuildConfig(), null!));
        }
    }
}
