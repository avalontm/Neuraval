using System;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class MultiHeadAttentionTensorMigrationTests
    {
        private const float Tolerance = 1e-3f;

        private static float[,] RandomMatrix(int rows, int cols, int seed)
        {
            var random = new Random(seed);
            var matrix = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                }
            }

            return matrix;
        }

        private static void AssertMatricesEqual(float[,] expected, float[,] actual)
        {
            Assert.Equal(expected.GetLength(0), actual.GetLength(0));
            Assert.Equal(expected.GetLength(1), actual.GetLength(1));

            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    Assert.True(MathF.Abs(expected[i, j] - actual[i, j]) < Tolerance,
                        $"Diferencia en [{i},{j}]: esperado {expected[i, j]}, obtenido {actual[i, j]}");
                }
            }
        }

        [Fact]
        public void Forward_IsDeterministic_AcrossRepeatedCalls()
        {
            var input = RandomMatrix(6, 8, 1);
            var attention = new MultiHeadAttention(embeddingDim: 8, numHeads: 2, seed: 42);

            var first = attention.Forward(input);
            var second = attention.Forward(input);

            AssertMatricesEqual(first, second);
        }

        [Fact]
        public void Forward_SingleHead_MatchesManualScaledDotProductAttention()
        {
            const int embeddingDim = 6;
            var input = RandomMatrix(5, embeddingDim, 2);
            var attention = new MultiHeadAttention(embeddingDim: embeddingDim, numHeads: 1, seed: 9);
            var state = attention.SaveState();

            var expected = ReferenceSingleHeadForward(input, state.QueryWeights, state.KeyWeights,
                state.ValueWeights, state.OutputWeights, embeddingDim);
            var actual = attention.Forward(input);

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void Forward_WithCausalMask_ZerosOutFutureAttentionWeights()
        {
            const int seqLen = 4;
            const int embeddingDim = 6;
            var input = RandomMatrix(seqLen, embeddingDim, 3);
            var attention = new MultiHeadAttention(embeddingDim: embeddingDim, numHeads: 2, seed: 5);

            var mask = new float[seqLen, seqLen];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < seqLen; j++)
                    mask[i, j] = j <= i ? 1f : 0f;

            attention.Forward(input, mask);

            var attentionWeightsField = attention.GetType()
                .GetField("_lastAttentionWeights", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var attentionWeights = (float[,,])attentionWeightsField!.GetValue(attention)!;

            for (int head = 0; head < 2; head++)
                for (int i = 0; i < seqLen; i++)
                    for (int j = i + 1; j < seqLen; j++)
                        Assert.True(attentionWeights[head, i, j] < 1e-6f);
        }

        [Fact]
        public void ForwardBatch_MatchesForwardPerRow()
        {
            const int batchSize = 2;
            const int seqLen = 4;
            const int embeddingDim = 6;

            var batch = new float[batchSize, seqLen, embeddingDim];
            var random = new Random(6);

            for (int b = 0; b < batchSize; b++)
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < embeddingDim; j++)
                        batch[b, i, j] = (float)(random.NextDouble() * 2.0 - 1.0);

            var attention = new MultiHeadAttention(embeddingDim: embeddingDim, numHeads: 2, seed: 13);

            var singleInput0 = new float[seqLen, embeddingDim];
            var singleInput1 = new float[seqLen, embeddingDim];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < embeddingDim; j++)
                {
                    singleInput0[i, j] = batch[0, i, j];
                    singleInput1[i, j] = batch[1, i, j];
                }

            var expected0 = attention.Forward(singleInput0);
            var expected1 = attention.Forward(singleInput1);

            var actualBatch = attention.ForwardBatch(batch, (float[,]?)null);

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    Assert.True(MathF.Abs(expected0[i, j] - actualBatch[0, i, j]) < Tolerance);
                    Assert.True(MathF.Abs(expected1[i, j] - actualBatch[1, i, j]) < Tolerance);
                }
            }
        }

        [Fact]
        public void Forward_ThenBackward_ProducesNonZeroGradients()
        {
            var input = RandomMatrix(4, 6, 7);
            var gradOutput = RandomMatrix(4, 6, 8);
            var attention = new MultiHeadAttention(embeddingDim: 6, numHeads: 2, seed: 17);

            attention.Forward(input);
            var gradInput = attention.Backward(gradOutput, 0.001f);
            attention.AverageGradients(1);

            Assert.Equal(4, gradInput.GetLength(0));
            Assert.Equal(6, gradInput.GetLength(1));
            Assert.True(attention.SumSquaredGradients() > 0f);
        }

        [Fact]
        public void Backward_MatchesManualOracle()
        {
            const int embeddingDim = 6;
            const int numHeads = 2;
            const int seqLen = 5;

            var input = RandomMatrix(seqLen, embeddingDim, 31);
            var gradOutput = RandomMatrix(seqLen, embeddingDim, 32);
            var attention = new MultiHeadAttention(embeddingDim: embeddingDim, numHeads: numHeads, seed: 13);

            attention.Forward(input);
            var gradInput = attention.Backward(gradOutput, 0.001f);
            attention.AverageGradients(1);

            var state = attention.SaveState();
            var queryWeights = Unflatten(state.QueryWeights, embeddingDim, embeddingDim);
            var keyWeights = Unflatten(state.KeyWeights, embeddingDim, embeddingDim);
            var valueWeights = Unflatten(state.ValueWeights, embeddingDim, embeddingDim);
            var outputWeights = Unflatten(state.OutputWeights, embeddingDim, embeddingDim);

            var oracle = ManualMultiHeadAttentionBackward(input, gradOutput, queryWeights, keyWeights,
                valueWeights, outputWeights, embeddingDim, numHeads);

            AssertMatricesEqual(oracle.GradInput, gradInput);

            for (int i = 0; i < embeddingDim; i++)
                for (int j = 0; j < embeddingDim; j++)
                {
                    Assert.True(MathF.Abs(oracle.QueryGrad[i, j] - attention.QueryGradientAt(i, j)) < Tolerance);
                    Assert.True(MathF.Abs(oracle.KeyGrad[i, j] - attention.KeyGradientAt(i, j)) < Tolerance);
                    Assert.True(MathF.Abs(oracle.ValueGrad[i, j] - attention.ValueGradientAt(i, j)) < Tolerance);
                    Assert.True(MathF.Abs(oracle.OutputGrad[i, j] - attention.OutputGradientAt(i, j)) < Tolerance);
                }
        }

        [Fact]
        public void BackwardBatch_MatchesBackwardPerSequence()
        {
            const int batchSize = 2;
            const int seqLen = 4;
            const int embeddingDim = 6;
            const int numHeads = 2;

            var batchInput = new float[batchSize, seqLen, embeddingDim];
            var batchGradOutput = new float[batchSize, seqLen, embeddingDim];
            var random = new Random(41);

            for (int b = 0; b < batchSize; b++)
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        batchInput[b, i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                        batchGradOutput[b, i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                    }

            var referenceAttention = new MultiHeadAttention(embeddingDim: embeddingDim, numHeads: numHeads, seed: 55);
            var expectedGradInput = new float[batchSize, seqLen, embeddingDim];

            for (int b = 0; b < batchSize; b++)
            {
                var singleInput = new float[seqLen, embeddingDim];
                var singleGradOutput = new float[seqLen, embeddingDim];
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        singleInput[i, j] = batchInput[b, i, j];
                        singleGradOutput[i, j] = batchGradOutput[b, i, j];
                    }

                referenceAttention.Forward(singleInput);
                var singleGradInput = referenceAttention.Backward(singleGradOutput, 0.001f);

                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < embeddingDim; j++)
                        expectedGradInput[b, i, j] = singleGradInput[i, j];
            }
            referenceAttention.AverageGradients(batchSize);

            var batchAttention = new MultiHeadAttention(embeddingDim: embeddingDim, numHeads: numHeads, seed: 55);
            batchAttention.ForwardBatch(batchInput, (float[,]?)null);
            var actualGradInput = batchAttention.BackwardBatch(batchGradOutput, 0.001f);
            batchAttention.AverageGradients(batchSize);

            for (int b = 0; b < batchSize; b++)
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < embeddingDim; j++)
                        Assert.True(MathF.Abs(expectedGradInput[b, i, j] - actualGradInput[b, i, j]) < Tolerance);

            for (int i = 0; i < embeddingDim; i++)
                for (int j = 0; j < embeddingDim; j++)
                {
                    Assert.True(MathF.Abs(referenceAttention.QueryGradientAt(i, j) - batchAttention.QueryGradientAt(i, j)) < Tolerance);
                    Assert.True(MathF.Abs(referenceAttention.KeyGradientAt(i, j) - batchAttention.KeyGradientAt(i, j)) < Tolerance);
                    Assert.True(MathF.Abs(referenceAttention.ValueGradientAt(i, j) - batchAttention.ValueGradientAt(i, j)) < Tolerance);
                    Assert.True(MathF.Abs(referenceAttention.OutputGradientAt(i, j) - batchAttention.OutputGradientAt(i, j)) < Tolerance);
                }
        }

        private static float[,] Unflatten(float[] flat, int rows, int cols)
        {
            var matrix = new float[rows, cols];
            int index = 0;

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = flat[index++];

            return matrix;
        }

        private readonly struct ManualBackwardResult
        {
            public ManualBackwardResult(float[,] gradInput, float[,] queryGrad, float[,] keyGrad, float[,] valueGrad, float[,] outputGrad)
            {
                GradInput = gradInput;
                QueryGrad = queryGrad;
                KeyGrad = keyGrad;
                ValueGrad = valueGrad;
                OutputGrad = outputGrad;
            }

            public float[,] GradInput { get; }
            public float[,] QueryGrad { get; }
            public float[,] KeyGrad { get; }
            public float[,] ValueGrad { get; }
            public float[,] OutputGrad { get; }
        }

        private static ManualBackwardResult ManualMultiHeadAttentionBackward(float[,] input, float[,] gradOutput,
            float[,] queryWeights, float[,] keyWeights, float[,] valueWeights, float[,] outputWeights,
            int embeddingDim, int numHeads)
        {
            int seqLen = input.GetLength(0);
            int headDim = embeddingDim / numHeads;
            float scale = MathF.Sqrt(headDim);

            var queries = MatMul(input, queryWeights);
            var keys = MatMul(input, keyWeights);
            var values = MatMul(input, valueWeights);

            var attentionWeights = new float[numHeads, seqLen, seqLen];
            var concatOutput = new float[seqLen, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                for (int i = 0; i < seqLen; i++)
                {
                    var scores = new float[seqLen];
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < seqLen; j++)
                    {
                        float dot = 0f;
                        for (int p = 0; p < headDim; p++)
                            dot += queries[i, startIdx + p] * keys[j, startIdx + p];
                        scores[j] = dot / scale;
                        max = MathF.Max(max, scores[j]);
                    }

                    float sum = 0f;
                    for (int j = 0; j < seqLen; j++)
                    {
                        scores[j] = MathF.Exp(scores[j] - max);
                        sum += scores[j];
                    }
                    for (int j = 0; j < seqLen; j++)
                    {
                        scores[j] /= sum;
                        attentionWeights[head, i, j] = scores[j];
                    }

                    for (int p = 0; p < headDim; p++)
                    {
                        float weightedSum = 0f;
                        for (int j = 0; j < seqLen; j++)
                            weightedSum += scores[j] * values[j, startIdx + p];
                        concatOutput[i, startIdx + p] = weightedSum;
                    }
                }
            }

            var gradConcatOutput = MatMulTransposeB(gradOutput, outputWeights);
            var outputGrad = MatMulTransposeA(concatOutput, gradOutput);

            var gradQueries = new float[seqLen, embeddingDim];
            var gradKeys = new float[seqLen, embeddingDim];
            var gradValues = new float[seqLen, embeddingDim];

            for (int head = 0; head < numHeads; head++)
            {
                int startIdx = head * headDim;

                for (int i = 0; i < seqLen; i++)
                {
                    var gradAttentionWeightsRow = new float[seqLen];
                    for (int j = 0; j < seqLen; j++)
                    {
                        float dot = 0f;
                        for (int p = 0; p < headDim; p++)
                            dot += gradConcatOutput[i, startIdx + p] * values[j, startIdx + p];
                        gradAttentionWeightsRow[j] = dot;
                    }

                    float weightedDot = 0f;
                    for (int k = 0; k < seqLen; k++)
                        weightedDot += attentionWeights[head, i, k] * gradAttentionWeightsRow[k];

                    var gradRawDotRow = new float[seqLen];
                    for (int j = 0; j < seqLen; j++)
                        gradRawDotRow[j] = attentionWeights[head, i, j] * (gradAttentionWeightsRow[j] - weightedDot) / scale;

                    for (int p = 0; p < headDim; p++)
                    {
                        float sumQ = 0f;
                        for (int j = 0; j < seqLen; j++)
                            sumQ += gradRawDotRow[j] * keys[j, startIdx + p];
                        gradQueries[i, startIdx + p] = sumQ;
                    }

                    for (int j = 0; j < seqLen; j++)
                    {
                        for (int p = 0; p < headDim; p++)
                        {
                            gradKeys[j, startIdx + p] += gradRawDotRow[j] * queries[i, startIdx + p];
                            gradValues[j, startIdx + p] += attentionWeights[head, i, j] * gradConcatOutput[i, startIdx + p];
                        }
                    }
                }
            }

            var queryGrad = MatMulTransposeA(input, gradQueries);
            var keyGrad = MatMulTransposeA(input, gradKeys);
            var valueGrad = MatMulTransposeA(input, gradValues);

            var gradInput = MatMulTransposeB(gradQueries, queryWeights);
            AddInPlace(gradInput, MatMulTransposeB(gradKeys, keyWeights));
            AddInPlace(gradInput, MatMulTransposeB(gradValues, valueWeights));

            return new ManualBackwardResult(gradInput, queryGrad, keyGrad, valueGrad, outputGrad);
        }

        private static float[,] MatMul(float[,] a, float[,] b)
        {
            int m = a.GetLength(0);
            int k = a.GetLength(1);
            int n = b.GetLength(1);
            var result = new float[m, n];

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < k; p++)
                        sum += a[i, p] * b[p, j];
                    result[i, j] = sum;
                }

            return result;
        }

        private static float[,] MatMulTransposeA(float[,] a, float[,] b)
        {
            int k = a.GetLength(0);
            int m = a.GetLength(1);
            int n = b.GetLength(1);
            var result = new float[m, n];

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < k; p++)
                        sum += a[p, i] * b[p, j];
                    result[i, j] = sum;
                }

            return result;
        }

        private static float[,] MatMulTransposeB(float[,] a, float[,] b)
        {
            int m = a.GetLength(0);
            int k = a.GetLength(1);
            int n = b.GetLength(0);
            var result = new float[m, n];

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < k; p++)
                        sum += a[i, p] * b[j, p];
                    result[i, j] = sum;
                }

            return result;
        }

        private static void AddInPlace(float[,] target, float[,] source)
        {
            int rows = target.GetLength(0);
            int cols = target.GetLength(1);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    target[i, j] += source[i, j];
        }

        private static float[,] ReferenceSingleHeadForward(float[,] input, float[] flatQueryWeights,
            float[] flatKeyWeights, float[] flatValueWeights, float[] flatOutputWeights, int embeddingDim)
        {
            int seqLen = input.GetLength(0);

            var queries = MatMul(input, flatQueryWeights, embeddingDim, embeddingDim);
            var keys = MatMul(input, flatKeyWeights, embeddingDim, embeddingDim);
            var values = MatMul(input, flatValueWeights, embeddingDim, embeddingDim);

            float scale = MathF.Sqrt(embeddingDim);
            var scores = new float[seqLen, seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    float dot = 0f;
                    for (int p = 0; p < embeddingDim; p++)
                    {
                        dot += queries[i, p] * keys[j, p];
                    }
                    scores[i, j] = dot / scale;
                }
            }

            var attentionWeights = new float[seqLen, seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < seqLen; j++)
                    max = MathF.Max(max, scores[i, j]);

                float sum = 0f;
                for (int j = 0; j < seqLen; j++)
                {
                    float exp = MathF.Exp(scores[i, j] - max);
                    attentionWeights[i, j] = exp;
                    sum += exp;
                }
                for (int j = 0; j < seqLen; j++)
                    attentionWeights[i, j] /= sum;
            }

            var concatOutput = new float[seqLen, embeddingDim];
            for (int i = 0; i < seqLen; i++)
            {
                for (int p = 0; p < embeddingDim; p++)
                {
                    float sum = 0f;
                    for (int j = 0; j < seqLen; j++)
                    {
                        sum += attentionWeights[i, j] * values[j, p];
                    }
                    concatOutput[i, p] = sum;
                }
            }

            return MatMul(concatOutput, flatOutputWeights, embeddingDim, embeddingDim);
        }

        private static float[,] MatMul(float[,] a, float[] flatB, int k, int n)
        {
            int m = a.GetLength(0);
            var result = new float[m, n];

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < k; p++)
                    {
                        sum += a[i, p] * flatB[p * n + j];
                    }
                    result[i, j] = sum;
                }
            }

            return result;
        }
    }
}
