using System;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class ModernDecoderBlockTests
    {
        private static TransformerConfig SmallConfig()
        {
            return new TransformerConfig
            {
                VocabSize = 32,
                HiddenSize = 8,
                NumHiddenLayers = 1,
                NumAttentionHeads = 4,
                NumKeyValueHeads = 2,
                IntermediateSize = 12,
                MaxPositionEmbeddings = 16,
                RopeTheta = 10000f,
                RmsNormEps = 1e-6f
            };
        }

        private static float[,,] RandomInput(int batch, int seq, int dim, int seed)
        {
            var rnd = new Random(seed);
            var input = new float[batch, seq, dim];
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < seq; s++)
                    for (int d = 0; d < dim; d++)
                        input[b, s, d] = (float)(rnd.NextDouble() * 2 - 1);
            return input;
        }

        [Fact]
        public void Forward_OutputShape_MatchesInputShape()
        {
            var config = SmallConfig();
            var block = new ModernDecoderBlock(config, seed: 1);
            var input = RandomInput(1, 5, config.HiddenSize, seed: 3);

            var output = block.Forward(input);

            Assert.Equal(input.GetLength(0), output.GetLength(0));
            Assert.Equal(input.GetLength(1), output.GetLength(1));
            Assert.Equal(input.GetLength(2), output.GetLength(2));
        }

        [Fact]
        public void Forward_DoesNotProduceNaN()
        {
            var config = SmallConfig();
            var block = new ModernDecoderBlock(config, seed: 2);
            var input = RandomInput(2, 4, config.HiddenSize, seed: 9);

            var output = block.Forward(input);

            for (int b = 0; b < output.GetLength(0); b++)
                for (int s = 0; s < output.GetLength(1); s++)
                    for (int d = 0; d < output.GetLength(2); d++)
                        Assert.False(float.IsNaN(output[b, s, d]));
        }

        [Fact]
        public void Forward_IsCausal_FutureTokenDoesNotAffectEarlierOutput()
        {
            var config = SmallConfig();
            var block = new ModernDecoderBlock(config, seed: 4);

            var inputA = RandomInput(1, 4, config.HiddenSize, seed: 11);
            var inputB = (float[,,])inputA.Clone();
            inputB[0, 3, 0] += 5.0f;

            var outputA = block.Forward(inputA);
            var outputB = block.Forward(inputB);

            for (int s = 0; s < 3; s++)
            {
                for (int d = 0; d < config.HiddenSize; d++)
                {
                    Assert.True(
                        MathF.Abs(outputA[0, s, d] - outputB[0, s, d]) < 1e-4f,
                        $"Un token futuro (posición 3) no debe afectar la salida de la posición {s}");
                }
            }
        }

        [Fact]
        public void Backward_InputGradient_MatchesNumericalGradient()
        {
            var config = SmallConfig();
            var block = new ModernDecoderBlock(config, seed: 6);
            var referenceState = block.SaveState();

            var input = RandomInput(1, 3, config.HiddenSize, seed: 13);
            var upstream = RandomInput(1, 3, config.HiddenSize, seed: 21);

            block.Forward(input);
            var analyticalGrad = block.Backward(upstream);

            float epsilon = 5e-3f;
            int seq = input.GetLength(1);
            int dim = config.HiddenSize;

            for (int s = 0; s < seq; s++)
            {
                for (int d = 0; d < dim; d++)
                {
                    var plus = (float[,,])input.Clone();
                    plus[0, s, d] += epsilon;
                    var minus = (float[,,])input.Clone();
                    minus[0, s, d] -= epsilon;

                    var blockPlus = ModernDecoderBlock.LoadState(referenceState);
                    var blockMinus = ModernDecoderBlock.LoadState(referenceState);

                    float lossPlus = Loss(blockPlus.Forward(plus), upstream);
                    float lossMinus = Loss(blockMinus.Forward(minus), upstream);

                    float numericalGrad = (lossPlus - lossMinus) / (2 * epsilon);

                    Assert.True(
                        MathF.Abs(numericalGrad - analyticalGrad[0, s, d]) < 5e-2f,
                        $"Mismatch at (0,{s},{d}): analytical={analyticalGrad[0, s, d]}, numerical={numericalGrad}");
                }
            }
        }

        [Fact]
        public void SaveState_LoadState_ProducesIdenticalForwardOutput()
        {
            var config = SmallConfig();
            var block = new ModernDecoderBlock(config, seed: 8);
            var input = RandomInput(1, 3, config.HiddenSize, seed: 15);

            var expected = block.Forward(input);

            var reloaded = ModernDecoderBlock.LoadState(block.SaveState());
            var actual = reloaded.Forward(input);

            for (int s = 0; s < 3; s++)
                for (int d = 0; d < config.HiddenSize; d++)
                    Assert.Equal(expected[0, s, d], actual[0, s, d], precision: 5);
        }

        private static float Loss(float[,,] output, float[,,] upstream)
        {
            float total = 0f;
            for (int b = 0; b < output.GetLength(0); b++)
                for (int s = 0; s < output.GetLength(1); s++)
                    for (int d = 0; d < output.GetLength(2); d++)
                        total += output[b, s, d] * upstream[b, s, d];
            return total;
        }
    }
}
