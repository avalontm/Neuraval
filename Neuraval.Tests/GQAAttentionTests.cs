using System;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class GQAAttentionTests
    {
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
        public void Forward_GroupedKeyValueHeads_AreSharedAcrossCorrectQueryHeads()
        {
            int hiddenSize = 8;
            int numAttentionHeads = 4;
            int numKeyValueHeads = 2;
            int headDim = hiddenSize / numAttentionHeads;

            var attention = new GQAAttention(hiddenSize, numAttentionHeads, numKeyValueHeads, seed: 1);
            var input = RandomInput(1, 3, hiddenSize, seed: 5);
            var baseline = attention.Forward(input);

            var state = attention.SaveState();
            state.Wk[0, 0] += 3.0f;
            var perturbed = GQAAttention.LoadState(state).Forward(input);

            bool group0Changed = false;
            bool group1Unchanged = true;

            for (int s = 0; s < 3; s++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    if (MathF.Abs(baseline[0, s, d] - perturbed[0, s, d]) > 1e-4f)
                        group0Changed = true;

                    int head3Offset = 3 * headDim + d;
                    if (MathF.Abs(baseline[0, s, head3Offset] - perturbed[0, s, head3Offset]) > 1e-4f)
                        group1Unchanged = false;
                }
            }

            Assert.True(group0Changed, "Perturbar Wk de kv-head 0 debe afectar a los query-heads de su grupo");
            Assert.True(group1Unchanged, "Perturbar Wk de kv-head 0 no debe afectar a query-heads de otro grupo");
        }

        [Fact]
        public void Forward_WithMultiQueryAttention_RunsWithSingleKeyValueHead()
        {
            var mqa = new GQAAttention(hiddenSize: 8, numAttentionHeads: 4, numKeyValueHeads: 1, seed: 2);
            var input = RandomInput(1, 3, 8, seed: 7);

            var output = mqa.Forward(input);

            Assert.Equal(3, output.GetLength(1));
            Assert.Equal(8, output.GetLength(2));
            for (int s = 0; s < 3; s++)
                for (int d = 0; d < 8; d++)
                    Assert.False(float.IsNaN(output[0, s, d]));
        }

        [Fact]
        public void Forward_IsCausal_FutureTokensDoNotAffectPastOutputs()
        {
            var attention = new GQAAttention(hiddenSize: 6, numAttentionHeads: 2, numKeyValueHeads: 2, seed: 3);

            var inputA = RandomInput(1, 4, 6, seed: 11);
            var inputB = (float[,,])inputA.Clone();
            inputB[0, 3, 2] += 5f;

            var outputA = attention.Forward(inputA);
            var outputB = attention.Forward(inputB);

            for (int s = 0; s < 3; s++)
                for (int d = 0; d < 6; d++)
                    Assert.Equal(outputA[0, s, d], outputB[0, s, d], precision: 4);
        }

        [Fact]
        public void Backward_InputGradient_MatchesNumericalGradient()
        {
            int hiddenSize = 4;
            int seqLen = 3;
            var attention = new GQAAttention(hiddenSize, numAttentionHeads: 2, numKeyValueHeads: 1, seed: 4);
            var input = RandomInput(1, seqLen, hiddenSize, seed: 13);
            var upstream = RandomInput(1, seqLen, hiddenSize, seed: 17);

            attention.Forward(input);
            var analyticalGrad = attention.Backward(upstream);

            float epsilon = 5e-3f;

            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < hiddenSize; d++)
                {
                    var plus = (float[,,])input.Clone();
                    plus[0, s, d] += epsilon;
                    var minus = (float[,,])input.Clone();
                    minus[0, s, d] -= epsilon;

                    float lossPlus = Loss(new GQAAttention(hiddenSize, 2, 1, seed: 4).Forward(plus), upstream);
                    float lossMinus = Loss(new GQAAttention(hiddenSize, 2, 1, seed: 4).Forward(minus), upstream);

                    float numericalGrad = (lossPlus - lossMinus) / (2 * epsilon);

                    Assert.True(
                        MathF.Abs(numericalGrad - analyticalGrad[0, s, d]) < 5e-2f,
                        $"Mismatch at ({s},{d}): analytical={analyticalGrad[0, s, d]}, numerical={numericalGrad}");
                }
            }
        }

        [Fact]
        public void Backward_OutputProjectionWeightGradient_MatchesNumericalGradient()
        {
            int hiddenSize = 4;
            int seqLen = 2;
            var input = RandomInput(1, seqLen, hiddenSize, seed: 21);
            var upstream = RandomInput(1, seqLen, hiddenSize, seed: 23);
            float epsilon = 5e-3f;

            var reference = new GQAAttention(hiddenSize, 2, 2, seed: 6);
            reference.Forward(input);
            reference.Backward(upstream);
            reference.AverageGradients(1);
            var state = reference.SaveState();

            for (int i = 0; i < 2; i++)
            {
                for (int o = 0; o < 2; o++)
                {
                    var plusState = CloneState(state);
                    plusState.Wo[i, o] += epsilon;
                    var attentionPlus = GQAAttention.LoadState(plusState);
                    float lossPlus = Loss(attentionPlus.Forward(input), upstream);

                    var minusState = CloneState(state);
                    minusState.Wo[i, o] -= epsilon;
                    var attentionMinus = GQAAttention.LoadState(minusState);
                    float lossMinus = Loss(attentionMinus.Forward(input), upstream);

                    float numericalGrad = (lossPlus - lossMinus) / (2 * epsilon);

                    Assert.True(
                        MathF.Abs(numericalGrad - AnalyticalWoGradient(reference, i, o)) < 1e-1f);
                }
            }
        }

        private static float AnalyticalWoGradient(GQAAttention attention, int i, int o)
        {
            var field = typeof(GQAAttention).GetField("_woGradients", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
            var gradients = (float[,])field.GetValue(attention)!;
            return gradients[i, o];
        }

        private static GQAAttentionState CloneState(GQAAttentionState state)
        {
            return new GQAAttentionState
            {
                HiddenSize = state.HiddenSize,
                NumAttentionHeads = state.NumAttentionHeads,
                NumKeyValueHeads = state.NumKeyValueHeads,
                Wq = (float[,])state.Wq.Clone(),
                Wk = (float[,])state.Wk.Clone(),
                Wv = (float[,])state.Wv.Clone(),
                Wo = (float[,])state.Wo.Clone()
            };
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
