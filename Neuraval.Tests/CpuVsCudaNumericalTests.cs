using System;
using Neuraval.Core.Utils;
using Neuraval.Cuda;
using Xunit;

namespace Neuraval.Tests
{
    public class CpuVsCudaNumericalTests
    {
        private const float Tolerance = 1e-3f;

        private static bool CudaReady()
        {
            if (!CudaDevice.IsAvailable())
            {
                return false;
            }

            if (!CudaDevice.Initialized)
            {
                CudaDevice.Initialize();
            }

            return true;
        }

        private static float[,] RandomMatrix(int rows, int cols, int seed)
        {
            var random = new Random(seed);
            var matrix = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
            return matrix;
        }

        private static void AssertMatricesClose(float[,] expected, float[,] actual, float tolerance, string label)
        {
            Assert.Equal(expected.GetLength(0), actual.GetLength(0));
            Assert.Equal(expected.GetLength(1), actual.GetLength(1));

            float maxAbsError = 0f;
            float sumAbsError = 0f;
            int count = 0;

            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    float diff = MathF.Abs(expected[i, j] - actual[i, j]);
                    maxAbsError = MathF.Max(maxAbsError, diff);
                    sumAbsError += diff;
                    count++;
                }
            }

            float meanAbsError = sumAbsError / count;
            Assert.True(maxAbsError < tolerance,
                $"{label}: maxAbsError={maxAbsError:E6} meanAbsError={meanAbsError:E6} excede tolerancia {tolerance:E6}");
        }

        [Fact]
        public void MatrixMultiply_CpuVsCuda()
        {
            if (!CudaReady()) return;

            var a = RandomMatrix(37, 53, 1);
            var b = RandomMatrix(53, 29, 2);

            var cpu = Matematicas.ParallelMatrixMultiply(a, b);
            var cuda = CudaMath.MatrixMultiply(a, b);

            AssertMatricesClose(cpu, cuda, Tolerance, nameof(MatrixMultiply_CpuVsCuda));
        }

        [Fact]
        public void MatrixMultiplyTransposeB_CpuVsCuda()
        {
            if (!CudaReady()) return;

            var a = RandomMatrix(41, 31, 3);
            var b = RandomMatrix(23, 31, 4);

            var cpu = Matematicas.ParallelMatrixMultiplyTransposeB(a, b, scale: 0.75f);
            var cuda = CudaMath.MatrixMultiplyTransposeB(a, b, scale: 0.75f);

            AssertMatricesClose(cpu, cuda, Tolerance, nameof(MatrixMultiplyTransposeB_CpuVsCuda));
        }

        [Fact]
        public void MatrixMultiplyTransposeA_CpuVsCuda()
        {
            if (!CudaReady()) return;

            var a = RandomMatrix(31, 19, 5);
            var b = RandomMatrix(31, 17, 6);

            var cpu = Matematicas.ParallelMatrixMultiplyTransposeA(a, b);
            var cuda = CudaMath.MatrixMultiplyTransposeA(a, b);

            AssertMatricesClose(cpu, cuda, Tolerance, nameof(MatrixMultiplyTransposeA_CpuVsCuda));
        }

        [Fact]
        public void SoftmaxRows_CpuVsCuda()
        {
            if (!CudaReady()) return;

            var input = RandomMatrix(17, 23, 7);

            var cpu = Matematicas.ParallelSoftmaxRows(input);
            var cuda = CudaMath.SoftmaxRows(input);

            AssertMatricesClose(cpu, cuda, Tolerance, nameof(SoftmaxRows_CpuVsCuda));

            for (int i = 0; i < cuda.GetLength(0); i++)
            {
                float rowSum = 0f;
                for (int j = 0; j < cuda.GetLength(1); j++)
                {
                    rowSum += cuda[i, j];
                }
                Assert.True(MathF.Abs(rowSum - 1.0f) < 1e-3f, $"Fila {i} de softmax CUDA no suma 1: {rowSum}");
            }
        }

        [Fact]
        public void LayerNormRows_CpuVsCuda()
        {
            if (!CudaReady()) return;

            var input = RandomMatrix(13, 11, 8);
            var gamma = new float[11];
            var beta = new float[11];
            for (int j = 0; j < 11; j++)
            {
                gamma[j] = 1.0f + j * 0.01f;
                beta[j] = j * 0.02f - 0.1f;
            }

            var cpu = Matematicas.ParallelLayerNormRows(input, gamma, beta, 1e-5f, out var cpuMean, out var cpuStd);
            var cuda = CudaMath.LayerNormRows(input, gamma, beta, 1e-5f, out var cudaMean, out var cudaStd);

            AssertMatricesClose(cpu, cuda, Tolerance, nameof(LayerNormRows_CpuVsCuda));

            for (int i = 0; i < cpuMean.Length; i++)
            {
                Assert.True(MathF.Abs(cpuMean[i] - cudaMean[i]) < Tolerance, $"mean[{i}] difiere: cpu={cpuMean[i]} cuda={cudaMean[i]}");
                Assert.True(MathF.Abs(cpuStd[i] - cudaStd[i]) < Tolerance, $"std[{i}] difiere: cpu={cpuStd[i]} cuda={cudaStd[i]}");
            }
        }

        [Fact]
        public void AdamUpdate_CpuVsCuda()
        {
            if (!CudaReady()) return;

            var paramsCpu = RandomMatrix(19, 23, 9);
            var gradsCpu = RandomMatrix(19, 23, 10);
            var mCpu = new float[19, 23];
            var vCpu = new float[19, 23];

            var paramsCuda = (float[,])paramsCpu.Clone();
            var gradsCuda = (float[,])gradsCpu.Clone();
            var mCuda = new float[19, 23];
            var vCuda = new float[19, 23];

            const float beta1 = 0.9f;
            const float beta2 = 0.999f;
            const float epsilon = 1e-8f;
            const float learningRate = 0.001f;
            const float biasCorrection1 = 1f - 0.9f;
            const float biasCorrection2 = 1f - 0.999f;

            Matematicas.ParallelAdamUpdate(paramsCpu, gradsCpu, mCpu, vCpu, beta1, beta2, epsilon, learningRate, biasCorrection1, biasCorrection2);
            CudaMath.AdamUpdate(paramsCuda, gradsCuda, mCuda, vCuda, beta1, beta2, epsilon, learningRate, biasCorrection1, biasCorrection2);

            AssertMatricesClose(paramsCpu, paramsCuda, Tolerance, nameof(AdamUpdate_CpuVsCuda) + ".params");
            AssertMatricesClose(mCpu, mCuda, Tolerance, nameof(AdamUpdate_CpuVsCuda) + ".m");
            AssertMatricesClose(vCpu, vCuda, Tolerance, nameof(AdamUpdate_CpuVsCuda) + ".v");
        }
    }
}
