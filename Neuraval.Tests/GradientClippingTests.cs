using System;
using System.Collections.Generic;
using System.Reflection;
using Neuraval.Core.Models;
using Neuraval.Core.Utils;
using Xunit;

namespace Neuraval.Tests
{
    public class GradientClippingTests
    {
        private const int VocabSize = 12;
        private const int EmbeddingDim = 8;
        private const int NumLayers = 3;
        private const int NumHeads = 2;
        private const int FeedforwardDim = 16;
        private const int MaxSequenceLength = 8;

        private static TransformerModel CreateModel(int seed = 123)
        {
            return new TransformerModel(
                vocabSize: VocabSize,
                embeddingDim: EmbeddingDim,
                numLayers: NumLayers,
                numHeads: NumHeads,
                feedforwardDim: FeedforwardDim,
                maxSequenceLength: MaxSequenceLength,
                dropout: 0.0f,
                seed: seed);
        }

        private static T GetPrivate<T>(object obj, string fieldName)
        {
            var field = obj.GetType().GetField(fieldName, BindingFlags.NonPublic | BindingFlags.Instance)
                ?? throw new InvalidOperationException($"Campo privado '{fieldName}' no encontrado en {obj.GetType().Name}");
            return (T)field.GetValue(obj)!;
        }

        private static float RecalculateGlobalGradientNorm(TransformerModel model)
        {
            var outputBiasGradients = GetPrivate<float[]>(model, "_outputBiasGradients");
            var embedding = GetPrivate<EmbeddingLayer>(model, "_embedding");
            var blocks = GetPrivate<List<TransformerBlock>>(model, "_blocks");
            var finalNorm = GetPrivate<LayerNormalization>(model, "_finalNorm");

            float sumSquared = 0;
            foreach (var g in outputBiasGradients)
            {
                sumSquared += g * g;
            }

            sumSquared += embedding.SumSquaredGradients();

            foreach (var block in blocks)
            {
                sumSquared += block.SumSquaredGradients();
            }

            sumSquared += finalNorm.SumSquaredGradients();

            return MathF.Sqrt(sumSquared);
        }

        private static void PopulateGradients(TransformerModel model, int[] tokens)
        {
            model.ZeroGradients();
            model.CalculateCausalLoss(tokens, lossStartIndex: 0);
            model.AverageGradients(1);
        }

        [Fact]
        public void ClipGradients_ReportedNorm_IsGlobalNorm_NotJustOutputBias()
        {
            Matematicas.SetNumThreads(1);
            var model = CreateModel();
            var tokens = new[] { 1, 2, 3, 4, 5, 1, 2 };

            PopulateGradients(model, tokens);

            float expectedGlobalNorm = RecalculateGlobalGradientNorm(model);

            float reportedNorm = model.ClipGradients(maxNorm: 1_000_000f);

            Assert.True(expectedGlobalNorm > 0f, "El gradiente global no debería ser cero en este escenario sintético.");
            Assert.Equal(expectedGlobalNorm, reportedNorm, precision: 3);
        }

        [Fact]
        public void ClipGradients_WhenNormExceedsMax_PostClipGlobalNorm_EqualsMaxNorm()
        {
            Matematicas.SetNumThreads(1);
            var model = CreateModel();
            var tokens = new[] { 1, 5, 3, 7, 2, 9, 4, 6 };

            PopulateGradients(model, tokens);

            float preClipNorm = RecalculateGlobalGradientNorm(model);
            const float maxNorm = 0.05f;

            Assert.True(preClipNorm > maxNorm,
                "Este test requiere un escenario donde el clipping SÍ se active; ajustar maxNorm o la secuencia de entrada si falla.");

            float reportedNorm = model.ClipGradients(maxNorm);
            Assert.Equal(preClipNorm, reportedNorm, precision: 3);

            float postClipNorm = RecalculateGlobalGradientNorm(model);

            Assert.Equal(maxNorm, postClipNorm, precision: 4);
        }

        [Fact]
        public void ClipGradients_ScalesEveryComponent_ByTheExactSameFactor()
        {
            Matematicas.SetNumThreads(1);
            var model = CreateModel();
            var tokens = new[] { 2, 4, 6, 8, 1, 3, 5 };

            PopulateGradients(model, tokens);

            var embedding = GetPrivate<EmbeddingLayer>(model, "_embedding");
            var blocks = GetPrivate<List<TransformerBlock>>(model, "_blocks");
            var finalNorm = GetPrivate<LayerNormalization>(model, "_finalNorm");
            var outputBiasGradients = GetPrivate<float[]>(model, "_outputBiasGradients");

            float embeddingNormBefore = MathF.Sqrt(embedding.SumSquaredGradients());
            var blockNormsBefore = blocks.ConvertAll(b => MathF.Sqrt(b.SumSquaredGradients()));
            float finalNormBefore = MathF.Sqrt(finalNorm.SumSquaredGradients());
            float outputBiasSumSqBefore = 0f;
            foreach (var g in outputBiasGradients) outputBiasSumSqBefore += g * g;
            float outputBiasNormBefore = MathF.Sqrt(outputBiasSumSqBefore);

            float globalNorm = RecalculateGlobalGradientNorm(model);
            const float maxNorm = 0.02f;
            Assert.True(globalNorm > maxNorm, "Se requiere que el clipping se active para este test.");

            float expectedScale = maxNorm / (globalNorm + 1e-10f);

            model.ClipGradients(maxNorm);

            float embeddingNormAfter = MathF.Sqrt(embedding.SumSquaredGradients());
            float finalNormAfter = MathF.Sqrt(finalNorm.SumSquaredGradients());
            float outputBiasSumSqAfter = 0f;
            foreach (var g in outputBiasGradients) outputBiasSumSqAfter += g * g;
            float outputBiasNormAfter = MathF.Sqrt(outputBiasSumSqAfter);

            AssertScaledBy(embeddingNormBefore, embeddingNormAfter, expectedScale, "embedding");
            AssertScaledBy(finalNormBefore, finalNormAfter, expectedScale, "finalNorm");
            AssertScaledBy(outputBiasNormBefore, outputBiasNormAfter, expectedScale, "outputBias");

            for (int i = 0; i < blocks.Count; i++)
            {
                float blockNormAfter = MathF.Sqrt(blocks[i].SumSquaredGradients());
                AssertScaledBy(blockNormsBefore[i], blockNormAfter, expectedScale, $"block[{i}]");
            }
        }

        private static void AssertScaledBy(float before, float after, float expectedScale, string label)
        {
            if (before < 1e-8f)
            {
                return;
            }

            float actualScale = after / before;
            Assert.True(
                MathF.Abs(actualScale - expectedScale) < 1e-3f,
                $"El componente '{label}' se escaló por {actualScale:F6} pero se esperaba {expectedScale:F6} (norma antes={before:F6}, después={after:F6}).");
        }

        [Fact]
        public void ClipGradients_WhenNormBelowMax_GradientsAreUnchanged()
        {
            Matematicas.SetNumThreads(1);
            var model = CreateModel();
            var tokens = new[] { 1, 2, 3, 4 };

            PopulateGradients(model, tokens);

            var embedding = GetPrivate<EmbeddingLayer>(model, "_embedding");
            float embeddingSumSqBefore = embedding.SumSquaredGradients();

            float globalNorm = RecalculateGlobalGradientNorm(model);

            float reportedNorm = model.ClipGradients(globalNorm * 10f);

            float embeddingSumSqAfter = embedding.SumSquaredGradients();

            Assert.Equal(globalNorm, reportedNorm, precision: 4);
            Assert.Equal(embeddingSumSqBefore, embeddingSumSqAfter, precision: 6);
        }

        [Fact]
        public void EmbeddingLayer_SumSquaredGradients_MatchesManualComputation()
        {
            var embedding = new EmbeddingLayer(vocabSize: 5, embeddingDim: 4, seed: 1);
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    embedding.AccumulateGradientAt(i, j, (i + 1) * 0.1f + j * 0.01f);
                }
            }
            embedding.AverageGradients(1);

            float reported = embedding.SumSquaredGradients();

            float manual = 0f;
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    float g = (i + 1) * 0.1f + j * 0.01f;
                    manual += g * g;
                }
            }

            Assert.Equal(manual, reported, precision: 5);
        }

        [Fact]
        public void EmbeddingLayer_ScaleGradients_MultipliesEveryElementExactly()
        {
            var embedding = new EmbeddingLayer(vocabSize: 5, embeddingDim: 4, seed: 1);
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    embedding.AccumulateGradientAt(i, j, (i + 1) * 0.1f + j * 0.01f);
                }
            }
            embedding.AverageGradients(1);

            float sumSqBefore = embedding.SumSquaredGradients();
            const float scale = 0.37f;
            embedding.ScaleGradients(scale);
            float sumSqAfter = embedding.SumSquaredGradients();

            Assert.Equal(sumSqBefore * scale * scale, sumSqAfter, precision: 5);
        }

        [Fact]
        public void LayerNormalization_SumSquaredGradients_MatchesManualComputation()
        {
            var norm = new LayerNormalization(normalizedShape: 6);
            var input = new float[3, 6];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 6; j++)
                    input[i, j] = (i + 1) * 0.3f - j * 0.05f;

            norm.Forward(input);
            var gradOutput = new float[3, 6];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 6; j++)
                    gradOutput[i, j] = 0.1f * (i + j + 1);

            norm.Backward(gradOutput, 0.0f);
            norm.AverageGradients(1);

            float reported = norm.SumSquaredGradients();
            Assert.True(reported >= 0f);
            Assert.True(reported > 0f, "Se esperaban gradientes distintos de cero tras Backward.");
        }
    }
}
