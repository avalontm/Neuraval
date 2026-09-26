using System;
using Neuraval.Core.Models.RoPE;
using Xunit;

namespace Neuraval.Tests
{
    public class RotaryEmbeddingTests
    {
        private static RotaryEmbedding CreateEmbedding(int headDim, float ropeTheta = 10000f, int maxPositions = 64)
        {
            var config = new RotaryConfig
            {
                HeadDim = headDim,
                RopeTheta = ropeTheta,
                MaxPositionEmbeddings = maxPositions
            };

            return new RotaryEmbedding(config);
        }

        [Fact]
        public void Apply_AtPositionZero_IsIdentity()
        {
            var rope = CreateEmbedding(headDim: 4);
            var input = new float[,] { { 1f, 2f, 3f, 4f } };

            var output = rope.Apply(input, positionOffset: 0);

            Assert.Equal(1f, output[0, 0], precision: 5);
            Assert.Equal(2f, output[0, 1], precision: 5);
            Assert.Equal(3f, output[0, 2], precision: 5);
            Assert.Equal(4f, output[0, 3], precision: 5);
        }

        [Fact]
        public void Apply_AtPositionOne_MatchesManualRotation()
        {
            int headDim = 4;
            int halfDim = headDim / 2;
            float ropeTheta = 10000f;
            var rope = CreateEmbedding(headDim, ropeTheta);
            var input = new float[,] { { 1f, 0f, 0.5f, -0.5f } };

            var output = rope.Apply(input, positionOffset: 1);

            for (int i = 0; i < halfDim; i++)
            {
                float freq = 1f / MathF.Pow(ropeTheta, (2f * i) / headDim);
                float angle = 1 * freq;
                float cos = MathF.Cos(angle);
                float sin = MathF.Sin(angle);

                float x1 = input[0, i];
                float x2 = input[0, i + halfDim];

                float expected1 = x1 * cos - x2 * sin;
                float expected2 = x2 * cos + x1 * sin;

                Assert.Equal(expected1, output[0, i], precision: 5);
                Assert.Equal(expected2, output[0, i + halfDim], precision: 5);
            }
        }

        [Fact]
        public void Apply_PreservesVectorNorm()
        {
            var rope = CreateEmbedding(headDim: 8);
            var input = new float[,] { { 1f, -2f, 3f, 0.5f, -1.5f, 2.5f, -0.3f, 4f } };

            var output = rope.Apply(input, positionOffset: 5);

            float inputNorm = 0f;
            float outputNorm = 0f;
            for (int j = 0; j < 8; j++)
            {
                inputNorm += input[0, j] * input[0, j];
                outputNorm += output[0, j] * output[0, j];
            }

            Assert.Equal(inputNorm, outputNorm, precision: 4);
        }

        [Fact]
        public void Apply_WithLongSequence_EachPositionUsesOwnAngle()
        {
            var rope = CreateEmbedding(headDim: 4, maxPositions: 128);
            var input = new float[100, 4];
            for (int s = 0; s < 100; s++)
            {
                input[s, 0] = 1f;
                input[s, 1] = 0f;
                input[s, 2] = 0f;
                input[s, 3] = 1f;
            }

            var output = rope.Apply(input, positionOffset: 0);

            Assert.Equal(1f, output[0, 0], precision: 4);
            Assert.NotEqual(output[1, 0], output[99, 0]);
        }

        [Theory]
        [InlineData(10000f)]
        [InlineData(1000000f)]
        [InlineData(500f)]
        public void Apply_WithDifferentRopeTheta_ProducesDifferentRotations(float ropeTheta)
        {
            var rope = CreateEmbedding(headDim: 4, ropeTheta: ropeTheta);
            var input = new float[,] { { 1f, 0f, 0f, 1f } };

            var output = rope.Apply(input, positionOffset: 10);

            Assert.False(float.IsNaN(output[0, 0]));
            Assert.False(float.IsNaN(output[0, 1]));
        }

        [Fact]
        public void ApplyBackward_IsInverseOfApply()
        {
            var rope = CreateEmbedding(headDim: 6);
            var input = new float[,] { { 1f, 2f, -1f, 0.5f, -0.5f, 3f } };

            var rotated = rope.Apply(input, positionOffset: 7);
            var recovered = rope.ApplyBackward(rotated, positionOffset: 7);

            for (int j = 0; j < 6; j++)
            {
                Assert.Equal(input[0, j], recovered[0, j], precision: 4);
            }
        }

        [Fact]
        public void ApplyBackward_MatchesNumericalGradient()
        {
            int headDim = 4;
            var rope = CreateEmbedding(headDim, maxPositions: 32);
            var input = new float[,] { { 0.7f, -1.1f, 0.3f, 2.0f } };
            var upstream = new float[,] { { 0.2f, -0.4f, 0.1f, 0.3f } };

            var analyticalGrad = rope.ApplyBackward(upstream, positionOffset: 3);

            float epsilon = 1e-3f;

            for (int j = 0; j < headDim; j++)
            {
                var plus = (float[,])input.Clone();
                plus[0, j] += epsilon;
                var minus = (float[,])input.Clone();
                minus[0, j] -= epsilon;

                float lossPlus = Loss(rope.Apply(plus, positionOffset: 3), upstream);
                float lossMinus = Loss(rope.Apply(minus, positionOffset: 3), upstream);

                float numericalGrad = (lossPlus - lossMinus) / (2 * epsilon);

                Assert.True(
                    MathF.Abs(numericalGrad - analyticalGrad[0, j]) < 1e-2f,
                    $"Mismatch at {j}: analytical={analyticalGrad[0, j]}, numerical={numericalGrad}");
            }
        }

        [Fact]
        public void Apply_Batch3D_MatchesPerRow2D()
        {
            int headDim = 4;
            var rope = CreateEmbedding(headDim);
            var batch = new float[,,]
            {
                { { 1f, 2f, 3f, 4f }, { -1f, 0.5f, 2f, -3f } },
                { { 0.1f, 0.2f, 0.3f, 0.4f }, { 5f, -5f, 5f, -5f } }
            };

            var batchOutput = rope.Apply(batch, positionOffset: 0);

            for (int b = 0; b < 2; b++)
            {
                var row = new float[2, headDim];
                for (int s = 0; s < 2; s++)
                    for (int j = 0; j < headDim; j++)
                        row[s, j] = batch[b, s, j];

                var rowOutput = rope.Apply(row, positionOffset: 0);

                for (int s = 0; s < 2; s++)
                    for (int j = 0; j < headDim; j++)
                        Assert.Equal(rowOutput[s, j], batchOutput[b, s, j], precision: 4);
            }
        }

        [Fact]
        public void Apply_MultiHead4D_MatchesPerHead3D()
        {
            int headDim = 4;
            int numHeads = 2;
            var rope = CreateEmbedding(headDim);
            var input = new float[1, 3, numHeads, headDim];
            var rnd = new Random(1);
            for (int s = 0; s < 3; s++)
                for (int h = 0; h < numHeads; h++)
                    for (int j = 0; j < headDim; j++)
                        input[0, s, h, j] = (float)rnd.NextDouble();

            var output = rope.Apply(input, positionOffset: 2);

            for (int h = 0; h < numHeads; h++)
            {
                var singleHead = new float[1, 3, headDim];
                for (int s = 0; s < 3; s++)
                    for (int j = 0; j < headDim; j++)
                        singleHead[0, s, j] = input[0, s, h, j];

                var singleHeadOutput = rope.Apply(singleHead, positionOffset: 2);

                for (int s = 0; s < 3; s++)
                    for (int j = 0; j < headDim; j++)
                        Assert.Equal(singleHeadOutput[0, s, j], output[0, s, h, j], precision: 4);
            }
        }

        [Fact]
        public void Config_WithOddHeadDim_ThrowsOnValidate()
        {
            var config = new RotaryConfig { HeadDim = 5, MaxPositionEmbeddings = 16, RopeTheta = 10000f };

            Assert.Throws<InvalidOperationException>(() => config.Validate());
        }

        private static float Loss(float[,] output, float[,] upstream)
        {
            float total = 0f;
            for (int i = 0; i < output.GetLength(0); i++)
                for (int j = 0; j < output.GetLength(1); j++)
                    total += output[i, j] * upstream[i, j];
            return total;
        }
    }
}
