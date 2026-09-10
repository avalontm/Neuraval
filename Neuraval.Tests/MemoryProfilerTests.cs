using System.Reflection;
using Neuraval.Core.Models;
using Neuraval.Core.Utils;
using Xunit;

namespace Neuraval.Tests
{
    public class MemoryProfilerTests
    {
        private const int VocabSize = 50;
        private const int EmbeddingDim = 16;
        private const int NumLayers = 3;
        private const int NumHeads = 4;
        private const int FeedforwardDim = 32;
        private const int BatchSize = 2;
        private const int SequenceLength = 10;

        private static ModelMemoryReport CreateReport()
        {
            return MemoryProfiler.Analyze(
                vocabSize: VocabSize,
                embeddingDim: EmbeddingDim,
                numLayers: NumLayers,
                numHeads: NumHeads,
                feedforwardDim: FeedforwardDim,
                batchSize: BatchSize,
                sequenceLength: SequenceLength);
        }

        private static T GetPrivate<T>(object obj, string fieldName)
        {
            var field = obj.GetType().GetField(fieldName, BindingFlags.NonPublic | BindingFlags.Instance)
                ?? throw new MissingFieldException(obj.GetType().Name, fieldName);
            return (T)field.GetValue(obj)!;
        }

        [Fact]
        public void EmbeddingParameterCount_MatchesHandComputedFormula()
        {
            var report = CreateReport();
            var embeddingComponent = report.Parameters[0];

            Assert.Equal("Embedding", embeddingComponent.Name);
            Assert.Equal((long)VocabSize * EmbeddingDim, embeddingComponent.ParameterCount);
        }

        [Fact]
        public void AttentionParameterCount_MatchesHandComputedFormula()
        {
            var report = CreateReport();
            var attentionComponent = report.Parameters[1];

            long expected = 4L * EmbeddingDim * EmbeddingDim * NumLayers;
            Assert.Equal(expected, attentionComponent.ParameterCount);
        }

        [Fact]
        public void FeedForwardParameterCount_MatchesHandComputedFormula()
        {
            var report = CreateReport();
            var feedforwardComponent = report.Parameters[2];

            long perBlock = (long)EmbeddingDim * FeedforwardDim + FeedforwardDim
                + (long)FeedforwardDim * EmbeddingDim + EmbeddingDim;
            long expected = perBlock * NumLayers;

            Assert.Equal(expected, feedforwardComponent.ParameterCount);
        }

        [Fact]
        public void LayerNormParameterCount_MatchesHandComputedFormula()
        {
            var report = CreateReport();
            var layerNormComponent = report.Parameters[3];
            var finalNormComponent = report.Parameters[4];

            Assert.Equal(4L * EmbeddingDim * NumLayers, layerNormComponent.ParameterCount);
            Assert.Equal(2L * EmbeddingDim, finalNormComponent.ParameterCount);
        }

        [Fact]
        public void OutputBiasParameterCount_MatchesVocabSize()
        {
            var report = CreateReport();
            var outputBiasComponent = report.Parameters[5];

            Assert.Equal(VocabSize, outputBiasComponent.ParameterCount);
        }

        [Fact]
        public void GradientBytes_AreTwiceWeightsBytes_ForEveryComponent()
        {
            var report = CreateReport();

            foreach (var component in report.Parameters)
            {
                Assert.Equal(component.WeightsBytes * 2, component.GradientBytes);
            }
        }

        [Fact]
        public void OptimizerStateBytes_AreTwiceWeightsBytes_ForEveryComponent()
        {
            var report = CreateReport();

            foreach (var component in report.Parameters)
            {
                Assert.Equal(component.WeightsBytes * 2, component.OptimizerStateBytes);
            }
        }

        [Fact]
        public void TotalParameterBytes_IsFiveTimesWeightsBytes()
        {
            var report = CreateReport();

            Assert.Equal(report.TotalWeightsBytes * 5, report.TotalParameterBytes);
        }

        [Fact]
        public void TotalParameterCount_MatchesActualEmbeddingArraySize()
        {
            var model = new TransformerModel(
                vocabSize: VocabSize,
                embeddingDim: EmbeddingDim,
                numLayers: NumLayers,
                numHeads: NumHeads,
                feedforwardDim: FeedforwardDim,
                maxSequenceLength: SequenceLength,
                dropout: 0.0f,
                seed: 5);

            var embeddingLayer = GetPrivate<EmbeddingLayer>(model, "_embedding");
            var embeddings = GetPrivate<float[,]>(embeddingLayer, "_embeddings");

            long actualParameterCount = (long)embeddings.GetLength(0) * embeddings.GetLength(1);

            var report = CreateReport();
            var embeddingComponent = report.Parameters[0];

            Assert.Equal(actualParameterCount, embeddingComponent.ParameterCount);
        }

        [Fact]
        public void ActivationBytes_ScaleLinearlyWithBatchSize()
        {
            var singleBatchReport = MemoryProfiler.Analyze(
                vocabSize: VocabSize,
                embeddingDim: EmbeddingDim,
                numLayers: NumLayers,
                numHeads: NumHeads,
                feedforwardDim: FeedforwardDim,
                batchSize: 1,
                sequenceLength: SequenceLength);

            var doubleBatchReport = MemoryProfiler.Analyze(
                vocabSize: VocabSize,
                embeddingDim: EmbeddingDim,
                numLayers: NumLayers,
                numHeads: NumHeads,
                feedforwardDim: FeedforwardDim,
                batchSize: 2,
                sequenceLength: SequenceLength);

            Assert.Equal(singleBatchReport.TotalActivationBytes * 2, doubleBatchReport.TotalActivationBytes);
        }

        [Fact]
        public void ActivationBytes_ScaleQuadraticallyWithSequenceLengthForAttentionScores()
        {
            var shortSeqReport = MemoryProfiler.Analyze(
                vocabSize: VocabSize,
                embeddingDim: EmbeddingDim,
                numLayers: NumLayers,
                numHeads: NumHeads,
                feedforwardDim: FeedforwardDim,
                batchSize: 1,
                sequenceLength: 4);

            var longSeqReport = MemoryProfiler.Analyze(
                vocabSize: VocabSize,
                embeddingDim: EmbeddingDim,
                numLayers: NumLayers,
                numHeads: NumHeads,
                feedforwardDim: FeedforwardDim,
                batchSize: 1,
                sequenceLength: 8);

            var shortScores = shortSeqReport.Activations[2];
            var longScores = longSeqReport.Activations[2];

            Assert.Equal("Attention scores por cabeza", shortScores.Name);
            Assert.Equal(shortScores.ElementCount * 4, longScores.ElementCount);
        }

        [Theory]
        [InlineData(512, "512 B")]
        [InlineData(2048, "2.00 KB")]
        [InlineData(5 * 1024 * 1024, "5.00 MB")]
        [InlineData(3L * 1024 * 1024 * 1024, "3.00 GB")]
        public void FormatBytes_ProducesExpectedUnit(long bytes, string expected)
        {
            Assert.Equal(expected, MemoryProfiler.FormatBytes(bytes));
        }
    }
}
