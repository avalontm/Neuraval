using System;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class SwiGLUFeedForwardTests
    {
        [Fact]
        public void Forward_WithKnownWeights_MatchesManualComputation()
        {
            var state = new SwiGLUFeedForwardState
            {
                EmbeddingDim = 2,
                HiddenDim = 2,
                WeightsGate = Flatten(new float[,] { { 1f, 0f }, { 0f, 1f } }),
                WeightsUp = Flatten(new float[,] { { 1f, 0f }, { 0f, 1f } }),
                WeightsDown = Flatten(new float[,] { { 1f, 0f }, { 0f, 1f } })
            };
            var ffn = SwiGLUFeedForward.LoadState(state);

            var input = new float[,] { { 0.5f, -0.5f } };
            var output = ffn.Forward(input);

            float silu0 = SiLU(0.5f);
            float silu1 = SiLU(-0.5f);
            float expected0 = silu0 * 0.5f;
            float expected1 = silu1 * -0.5f;

            Assert.Equal(expected0, output[0, 0], precision: 5);
            Assert.Equal(expected1, output[0, 1], precision: 5);
        }

        [Fact]
        public void Forward_DoesNotProduceNaN_ForVariedInputs()
        {
            var ffn = new SwiGLUFeedForward(6, 10, seed: 7);
            var input = new float[,]
            {
                { 1f, -2f, 3f, -4f, 5f, -6f },
                { 0.1f, 0.2f, -0.3f, 0.4f, -0.5f, 0.6f }
            };

            var output = ffn.Forward(input);

            for (int i = 0; i < output.GetLength(0); i++)
            {
                for (int j = 0; j < output.GetLength(1); j++)
                {
                    Assert.False(float.IsNaN(output[i, j]));
                }
            }
        }

        [Fact]
        public void Backward_InputGradient_MatchesNumericalGradient()
        {
            int embeddingDim = 4;
            int hiddenDim = 5;

            var input = new float[,]
            {
                { 0.5f, -1.2f, 2.3f, 0.1f },
                { 1.1f, 0.3f, -2.0f, 0.4f }
            };
            var upstream = new float[,]
            {
                { 0.3f, -0.1f, 0.7f, 0.2f },
                { -0.2f, 0.5f, 0.1f, -0.6f }
            };

            var ffn = new SwiGLUFeedForward(embeddingDim, hiddenDim, seed: 3);
            var referenceState = ffn.SaveState();

            ffn.Forward(input);
            var analyticalGrad = ffn.Backward(upstream, 0f);

            float epsilon = 1e-3f;

            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    var plus = (float[,])input.Clone();
                    plus[i, j] += epsilon;
                    var minus = (float[,])input.Clone();
                    minus[i, j] -= epsilon;

                    var ffnPlus = SwiGLUFeedForward.LoadState(referenceState);
                    var ffnMinus = SwiGLUFeedForward.LoadState(referenceState);

                    float lossPlus = Loss(ffnPlus.Forward(plus), upstream);
                    float lossMinus = Loss(ffnMinus.Forward(minus), upstream);

                    float numericalGrad = (lossPlus - lossMinus) / (2 * epsilon);

                    Assert.True(
                        MathF.Abs(numericalGrad - analyticalGrad[i, j]) < 2e-2f,
                        $"Mismatch at ({i},{j}): analytical={analyticalGrad[i, j]}, numerical={numericalGrad}");
                }
            }
        }

        [Fact]
        public void Backward_DownWeightGradient_MatchesNumericalGradient()
        {
            int embeddingDim = 3;
            int hiddenDim = 4;

            var input = new float[,] { { 0.6f, -0.3f, 1.2f } };
            var upstream = new float[,] { { 0.2f, -0.4f, 0.1f } };
            float epsilon = 1e-3f;

            var baseFfn = new SwiGLUFeedForward(embeddingDim, hiddenDim, seed: 5);
            var baseState = baseFfn.SaveState();

            baseFfn.Forward(input);
            baseFfn.Backward(upstream, 0f);
            baseFfn.AverageGradients(1);

            for (int i = 0; i < hiddenDim; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    var statePlus = CloneState(baseState);
                    statePlus.WeightsDown[i * embeddingDim + j] += epsilon;
                    var stateMinus = CloneState(baseState);
                    stateMinus.WeightsDown[i * embeddingDim + j] -= epsilon;

                    float lossPlus = Loss(SwiGLUFeedForward.LoadState(statePlus).Forward(input), upstream);
                    float lossMinus = Loss(SwiGLUFeedForward.LoadState(stateMinus).Forward(input), upstream);

                    float numericalGrad = (lossPlus - lossMinus) / (2 * epsilon);
                    float analyticalGrad = baseFfn.DownGradientAt(i, j);

                    Assert.True(
                        MathF.Abs(numericalGrad - analyticalGrad) < 2e-2f,
                        $"Down grad mismatch at ({i},{j}): analytical={analyticalGrad}, numerical={numericalGrad}");
                }
            }
        }

        [Fact]
        public void Backward_GateWeightGradient_MatchesNumericalGradient()
        {
            int embeddingDim = 3;
            int hiddenDim = 4;

            var input = new float[,] { { 0.6f, -0.3f, 1.2f } };
            var upstream = new float[,] { { 0.2f, -0.4f, 0.1f } };
            float epsilon = 1e-3f;

            var baseFfn = new SwiGLUFeedForward(embeddingDim, hiddenDim, seed: 9);
            var baseState = baseFfn.SaveState();

            baseFfn.Forward(input);
            baseFfn.Backward(upstream, 0f);
            baseFfn.AverageGradients(1);

            for (int i = 0; i < embeddingDim; i++)
            {
                for (int j = 0; j < hiddenDim; j++)
                {
                    var statePlus = CloneState(baseState);
                    statePlus.WeightsGate[i * hiddenDim + j] += epsilon;
                    var stateMinus = CloneState(baseState);
                    stateMinus.WeightsGate[i * hiddenDim + j] -= epsilon;

                    float lossPlus = Loss(SwiGLUFeedForward.LoadState(statePlus).Forward(input), upstream);
                    float lossMinus = Loss(SwiGLUFeedForward.LoadState(stateMinus).Forward(input), upstream);

                    float numericalGrad = (lossPlus - lossMinus) / (2 * epsilon);
                    float analyticalGrad = baseFfn.GateGradientAt(i, j);

                    Assert.True(
                        MathF.Abs(numericalGrad - analyticalGrad) < 2e-2f,
                        $"Gate grad mismatch at ({i},{j}): analytical={analyticalGrad}, numerical={numericalGrad}");
                }
            }
        }

        [Fact]
        public void ForwardBatch_MatchesForwardAppliedPerRow()
        {
            int embeddingDim = 4;
            int hiddenDim = 6;

            var batch = new float[,,]
            {
                { { 1f, 2f, 3f, 4f }, { -1f, 0.5f, 2f, -3f } },
                { { 0.1f, 0.2f, 0.3f, 0.4f }, { 5f, -5f, 5f, -5f } }
            };

            var ffnBatch = new SwiGLUFeedForward(embeddingDim, hiddenDim, seed: 11);
            var state = ffnBatch.SaveState();
            var ffnSingle = SwiGLUFeedForward.LoadState(state);

            var batchOutput = ffnBatch.ForwardBatch(batch);

            for (int b = 0; b < 2; b++)
            {
                for (int s = 0; s < 2; s++)
                {
                    var row = new float[,] { { batch[b, s, 0], batch[b, s, 1], batch[b, s, 2], batch[b, s, 3] } };
                    var rowOutput = ffnSingle.Forward(row);

                    for (int j = 0; j < embeddingDim; j++)
                    {
                        Assert.Equal(rowOutput[0, j], batchOutput[b, s, j], precision: 4);
                    }
                }
            }
        }

        [Fact]
        public void BackwardBatch_InputGradient_MatchesForwardBackwardPerRow()
        {
            int embeddingDim = 3;
            int hiddenDim = 4;

            var batch = new float[,,]
            {
                { { 0.5f, -1.2f, 2.3f } },
                { { 1.1f, 0.3f, -2.0f } }
            };
            var upstreamBatch = new float[,,]
            {
                { { 0.3f, -0.1f, 0.7f } },
                { { -0.2f, 0.5f, 0.1f } }
            };

            var ffnBatch = new SwiGLUFeedForward(embeddingDim, hiddenDim, seed: 13);
            var state = ffnBatch.SaveState();

            ffnBatch.ForwardBatch(batch);
            var gradBatch = ffnBatch.BackwardBatch(upstreamBatch, 0f);

            for (int b = 0; b < 2; b++)
            {
                var ffnSingle = SwiGLUFeedForward.LoadState(state);
                var row = new float[,] { { batch[b, 0, 0], batch[b, 0, 1], batch[b, 0, 2] } };
                var upstreamRow = new float[,] { { upstreamBatch[b, 0, 0], upstreamBatch[b, 0, 1], upstreamBatch[b, 0, 2] } };

                ffnSingle.Forward(row);
                var gradRow = ffnSingle.Backward(upstreamRow, 0f);

                for (int j = 0; j < embeddingDim; j++)
                {
                    Assert.Equal(gradRow[0, j], gradBatch[b, 0, j], precision: 4);
                }
            }
        }

        private static float SiLU(float x)
        {
            float sigmoid = 1.0f / (1.0f + MathF.Exp(-x));
            return x * sigmoid;
        }

        private static float Loss(float[,] output, float[,] upstream)
        {
            float total = 0f;
            for (int i = 0; i < output.GetLength(0); i++)
            {
                for (int j = 0; j < output.GetLength(1); j++)
                {
                    total += output[i, j] * upstream[i, j];
                }
            }
            return total;
        }

        private static float[] Flatten(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows * cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i * cols + j] = matrix[i, j];
            return result;
        }

        private static SwiGLUFeedForwardState CloneState(SwiGLUFeedForwardState state)
        {
            return new SwiGLUFeedForwardState
            {
                EmbeddingDim = state.EmbeddingDim,
                HiddenDim = state.HiddenDim,
                WeightsGate = (float[])state.WeightsGate.Clone(),
                WeightsUp = (float[])state.WeightsUp.Clone(),
                WeightsDown = (float[])state.WeightsDown.Clone()
            };
        }
    }
}
