using System;
using Neuraval.Cuda;

namespace Neuraval.Tensor.Backends.Cpu
{
    public sealed class CpuScalarBackend : ICpuTensorBackend
    {
        public Tensor MatMul(Tensor a, Tensor b, int m, int k, int n)
        {
            var result = new Tensor(new[] { m, n }, a.Device, a.DType);

            for (int i = 0; i < m; i++)
            {
                int aRowOffset = i * k;
                int resultRowOffset = i * n;

                for (int j = 0; j < n; j++)
                {
                    float sum = 0;

                    for (int p = 0; p < k; p++)
                    {
                        sum += a.Buffer[aRowOffset + p] * b.Buffer[p * n + j];
                    }

                    result.Buffer[resultRowOffset + j] = sum;
                }
            }

            return result;
        }

        public Tensor MatMulTransposeB(Tensor a, Tensor b, int m, int k, int n, float scale)
        {
            var result = new Tensor(new[] { m, n }, a.Device, a.DType);

            for (int i = 0; i < m; i++)
            {
                int aRowOffset = i * k;
                int resultRowOffset = i * n;

                for (int j = 0; j < n; j++)
                {
                    int bRowOffset = j * k;
                    float sum = 0;

                    for (int p = 0; p < k; p++)
                    {
                        sum += a.Buffer[aRowOffset + p] * b.Buffer[bRowOffset + p];
                    }

                    result.Buffer[resultRowOffset + j] = sum * scale;
                }
            }

            return result;
        }

        public Tensor MatMulTransposeA(Tensor a, Tensor b, int k, int m, int n)
        {
            var result = new Tensor(new[] { m, n }, a.Device, a.DType);

            for (int i = 0; i < m; i++)
            {
                int resultRowOffset = i * n;

                for (int j = 0; j < n; j++)
                {
                    float sum = 0;

                    for (int p = 0; p < k; p++)
                    {
                        sum += a.Buffer[p * m + i] * b.Buffer[p * n + j];
                    }

                    result.Buffer[resultRowOffset + j] = sum;
                }
            }

            return result;
        }

        public Tensor MatMulCachedB(Tensor a, float[,] weights, CudaWeightCache weightCache, int m, int k, int n)
        {
            var result = new Tensor(new[] { m, n }, a.Device, a.DType);

            for (int i = 0; i < m; i++)
            {
                int aRowOffset = i * k;
                int resultRowOffset = i * n;

                for (int j = 0; j < n; j++)
                {
                    float sum = 0;

                    for (int p = 0; p < k; p++)
                    {
                        sum += a.Buffer[aRowOffset + p] * weights[p, j];
                    }

                    result.Buffer[resultRowOffset + j] = sum;
                }
            }

            return result;
        }

        public Tensor MatMulTransposeBCachedB(Tensor a, float[,] weights, CudaWeightCache weightCache, int m, int k, int n, float scale)
        {
            var result = new Tensor(new[] { m, n }, a.Device, a.DType);

            for (int i = 0; i < m; i++)
            {
                int aRowOffset = i * k;
                int resultRowOffset = i * n;

                for (int j = 0; j < n; j++)
                {
                    float sum = 0;

                    for (int p = 0; p < k; p++)
                    {
                        sum += a.Buffer[aRowOffset + p] * weights[j, p];
                    }

                    result.Buffer[resultRowOffset + j] = sum * scale;
                }
            }

            return result;
        }

        public Tensor Add(Tensor a, Tensor b)
        {
            var result = new Tensor((int[])a.Shape.Clone(), a.Device, a.DType);

            for (int i = 0; i < a.Buffer.Length; i++)
            {
                result.Buffer[i] = a.Buffer[i] + b.Buffer[i];
            }

            return result;
        }

        public void AddInPlace(Tensor target, Tensor source, float scale)
        {
            for (int i = 0; i < target.Buffer.Length; i++)
            {
                target.Buffer[i] += source.Buffer[i] * scale;
            }
        }

        public Tensor Scale(Tensor a, float scalar)
        {
            var result = new Tensor((int[])a.Shape.Clone(), a.Device, a.DType);

            for (int i = 0; i < a.Buffer.Length; i++)
            {
                result.Buffer[i] = a.Buffer[i] * scalar;
            }

            return result;
        }

        public Tensor Transpose(Tensor a, int rows, int cols)
        {
            var result = new Tensor(new[] { cols, rows }, a.Device, a.DType);

            for (int i = 0; i < rows; i++)
            {
                int aRowOffset = i * cols;

                for (int j = 0; j < cols; j++)
                {
                    result.Buffer[j * rows + i] = a.Buffer[aRowOffset + j];
                }
            }

            return result;
        }

        public Tensor ReLU(Tensor a)
        {
            var result = new Tensor((int[])a.Shape.Clone(), a.Device, a.DType);

            for (int i = 0; i < a.Buffer.Length; i++)
            {
                result.Buffer[i] = MathF.Max(0.0f, a.Buffer[i]);
            }

            return result;
        }

        public Tensor ReLUBackward(Tensor grad, Tensor activation)
        {
            var result = new Tensor((int[])grad.Shape.Clone(), grad.Device, grad.DType);

            for (int i = 0; i < grad.Buffer.Length; i++)
            {
                result.Buffer[i] = activation.Buffer[i] > 0.0f ? grad.Buffer[i] : 0.0f;
            }

            return result;
        }

        public Tensor Multiply(Tensor a, Tensor b)
        {
            var result = new Tensor((int[])a.Shape.Clone(), a.Device, a.DType);

            for (int i = 0; i < a.Buffer.Length; i++)
            {
                result.Buffer[i] = a.Buffer[i] * b.Buffer[i];
            }

            return result;
        }

        public Tensor SumRows(Tensor a, int rows, int cols)
        {
            var result = new Tensor(new[] { cols }, a.Device, a.DType);

            for (int j = 0; j < cols; j++)
            {
                float sum = 0;

                for (int i = 0; i < rows; i++)
                {
                    sum += a.Buffer[i * cols + j];
                }

                result.Buffer[j] = sum;
            }

            return result;
        }

        public float Sum(Tensor a)
        {
            float total = 0;

            for (int i = 0; i < a.Buffer.Length; i++)
            {
                total += a.Buffer[i];
            }

            return total;
        }

        public Tensor SoftmaxRows(Tensor a, int rows, int cols)
        {
            var result = new Tensor(new[] { rows, cols }, a.Device, a.DType);

            for (int i = 0; i < rows; i++)
            {
                int rowOffset = i * cols;
                float max = float.NegativeInfinity;

                for (int j = 0; j < cols; j++)
                {
                    max = MathF.Max(max, a.Buffer[rowOffset + j]);
                }

                float sum = 0;

                for (int j = 0; j < cols; j++)
                {
                    float exp = MathF.Exp(a.Buffer[rowOffset + j] - max);
                    result.Buffer[rowOffset + j] = exp;
                    sum += exp;
                }

                for (int j = 0; j < cols; j++)
                {
                    result.Buffer[rowOffset + j] /= sum;
                }
            }

            return result;
        }

        public Tensor SoftmaxRowsBackward(Tensor gradOutput, Tensor softmaxOutput, int rows, int cols)
        {
            var result = new Tensor(new[] { rows, cols }, gradOutput.Device, gradOutput.DType);

            for (int i = 0; i < rows; i++)
            {
                int rowOffset = i * cols;
                float dot = 0;

                for (int j = 0; j < cols; j++)
                {
                    dot += softmaxOutput.Buffer[rowOffset + j] * gradOutput.Buffer[rowOffset + j];
                }

                for (int j = 0; j < cols; j++)
                {
                    result.Buffer[rowOffset + j] = softmaxOutput.Buffer[rowOffset + j] * (gradOutput.Buffer[rowOffset + j] - dot);
                }
            }

            return result;
        }

        public Tensor LayerNormRows(Tensor a, Tensor gamma, Tensor beta, int rows, int cols, float epsilon, out float[] mean, out float[] std)
        {
            var result = new Tensor(new[] { rows, cols }, a.Device, a.DType);
            var meanBuffer = new float[rows];
            var stdBuffer = new float[rows];

            for (int i = 0; i < rows; i++)
            {
                int rowOffset = i * cols;
                float sum = 0;

                for (int j = 0; j < cols; j++)
                {
                    sum += a.Buffer[rowOffset + j];
                }

                float rowMean = sum / cols;
                float sumSquaredDiff = 0;

                for (int j = 0; j < cols; j++)
                {
                    float diff = a.Buffer[rowOffset + j] - rowMean;
                    sumSquaredDiff += diff * diff;
                }

                float rowStd = MathF.Sqrt(sumSquaredDiff / cols + epsilon);
                meanBuffer[i] = rowMean;
                stdBuffer[i] = rowStd;

                for (int j = 0; j < cols; j++)
                {
                    float normalized = (a.Buffer[rowOffset + j] - rowMean) / rowStd;
                    result.Buffer[rowOffset + j] = normalized * gamma.Buffer[j] + beta.Buffer[j];
                }
            }

            mean = meanBuffer;
            std = stdBuffer;

            return result;
        }
    }
}
