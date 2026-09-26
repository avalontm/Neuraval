using System;
using System.Collections.Generic;
using Neuraval.Core.Generation;
using Xunit;

namespace Neuraval.Tests
{
    public class TokenSamplerTests
    {
        [Fact]
        public void SampleNext_Greedy_ReturnsArgMaxRegardlessOfRandom()
        {
            var logits = new float[] { 1f, 5f, 2f, 4f };
            var options = GenerationOptions.CreateGreedy(10, Array.Empty<int>());

            var token = TokenSampler.SampleNext(logits, options, new List<int>(), new Random(1));

            Assert.Equal(1, token);
        }

        [Fact]
        public void SampleNext_TopKOne_AlwaysReturnsHighestLogitToken()
        {
            var logits = new float[] { 1f, 2f, 9f, 3f };
            var options = GenerationOptions.Create(false, 0.7f, 1f, 1, 1f, 10, Array.Empty<int>());

            for (int seed = 0; seed < 20; seed++)
            {
                var token = TokenSampler.SampleNext(logits, options, new List<int>(), new Random(seed));
                Assert.Equal(2, token);
            }
        }

        [Fact]
        public void SampleNext_RepetitionPenaltyDemotesRepeatedTopToken()
        {
            var logits = new float[] { 5f, 4f, 0f, 0f };
            var generatedTokenIds = new List<int> { 0 };
            var options = GenerationOptions.Create(false, 1f, 1f, 1, 10f, 10, Array.Empty<int>());

            var token = TokenSampler.SampleNext(logits, options, generatedTokenIds, new Random(1));

            Assert.Equal(1, token);
        }

        [Fact]
        public void SampleNext_Greedy_AlsoAppliesRepetitionPenalty()
        {
            // Regresión: antes el modo greedy ignoraba options.RepetitionPenalty y
            // devolvía siempre el token de máxima probabilidad cruda, aunque ya se
            // hubiera generado antes. Eso hace que una generación greedy quede
            // atrapada repitiendo el mismo fragmento para siempre en cuanto entra
            // en un ciclo de 1-2 tokens de alta probabilidad.
            var logits = new float[] { 5f, 4f, 0f, 0f };
            var generatedTokenIds = new List<int> { 0 };
            var options = GenerationOptions.Create(true, 1f, 1f, int.MaxValue, 10f, 10, Array.Empty<int>());

            var token = TokenSampler.SampleNext(logits, options, generatedTokenIds, new Random(1));

            Assert.Equal(1, token);
        }

        [Fact]
        public void SampleNext_Greedy_NoRepetitionPenalty_ReturnsArgMaxEvenIfAlreadyGenerated()
        {
            // Con penalty == 1f (sin penalización), el comportamiento previo se
            // conserva: greedy sigue devolviendo el argmax crudo.
            var logits = new float[] { 5f, 4f, 0f, 0f };
            var generatedTokenIds = new List<int> { 0 };
            var options = GenerationOptions.Create(true, 1f, 1f, int.MaxValue, 1f, 10, Array.Empty<int>());

            var token = TokenSampler.SampleNext(logits, options, generatedTokenIds, new Random(1));

            Assert.Equal(0, token);
        }

        [Fact]
        public void SampleNext_NullLogits_Throws()
        {
            var options = GenerationOptions.CreateGreedy(10, Array.Empty<int>());

            Assert.Throws<ArgumentNullException>(() =>
                TokenSampler.SampleNext(null!, options, new List<int>(), new Random(1)));
        }

        [Fact]
        public void SampleNext_NullOptions_Throws()
        {
            Assert.Throws<ArgumentNullException>(() =>
                TokenSampler.SampleNext(new float[] { 1f }, null!, new List<int>(), new Random(1)));
        }

        [Fact]
        public void SampleNext_NullRandom_Throws()
        {
            var options = GenerationOptions.CreateGreedy(10, Array.Empty<int>());

            Assert.Throws<ArgumentNullException>(() =>
                TokenSampler.SampleNext(new float[] { 1f }, options, new List<int>(), null!));
        }
    }
}
