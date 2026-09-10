using System;
using Neuraval.Cuda;
using Neuraval.Tensor;
using Xunit;

namespace Neuraval.Tests
{
    public class TensorOpsCudaTests
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

        private static float[] RandomBuffer(int length, int seed)
        {
            var random = new Random(seed);
            var buffer = new float[length];

            for (int i = 0; i < length; i++)
            {
                buffer[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }

            return buffer;
        }

        private static Tensor.Tensor CpuTensor(float[] buffer, params int[] shape)
        {
            return new Tensor.Tensor((float[])buffer.Clone(), shape, DeviceType.Cpu);
        }

        private static Tensor.Tensor CudaTensor(float[] buffer, params int[] shape)
        {
            return new Tensor.Tensor((float[])buffer.Clone(), shape, DeviceType.Cuda);
        }

        private static void AssertBuffersClose(float[] expected, float[] actual, string label)
        {
            Assert.Equal(expected.Length, actual.Length);

            float maxAbsError = 0f;

            for (int i = 0; i < expected.Length; i++)
            {
                maxAbsError = MathF.Max(maxAbsError, MathF.Abs(expected[i] - actual[i]));
            }

            Assert.True(maxAbsError < Tolerance, $"{label}: maxAbsError={maxAbsError:E6} excede tolerancia {Tolerance:E6}");
        }

        [Fact]
        public void MatMul_CpuVsCuda()
        {
            if (!CudaReady()) return;

            var bufferA = RandomBuffer(37 * 53, 1);
            var bufferB = RandomBuffer(53 * 29, 2);

            var cpu = TensorOps.MatMul(CpuTensor(bufferA, 37, 53), CpuTensor(bufferB, 53, 29));
            var cuda = TensorOps.MatMul(CudaTensor(bufferA, 37, 53), CudaTensor(bufferB, 53, 29));

            Assert.Equal(DeviceType.Cuda, cuda.Device);
            AssertBuffersClose(cpu.Buffer, cuda.Buffer, nameof(MatMul_CpuVsCuda));
        }

        [Fact]
        public void MatMulTransposeB_CpuVsCuda()
        {
            if (!CudaReady()) return;

            var bufferA = RandomBuffer(41 * 31, 3);
            var bufferB = RandomBuffer(23 * 31, 4);

            var cpu = TensorOps.MatMulTransposeB(CpuTensor(bufferA, 41, 31), CpuTensor(bufferB, 23, 31), 0.75f);
            var cuda = TensorOps.MatMulTransposeB(CudaTensor(bufferA, 41, 31), CudaTensor(bufferB, 23, 31), 0.75f);

            AssertBuffersClose(cpu.Buffer, cuda.Buffer, nameof(MatMulTransposeB_CpuVsCuda));
        }

        [Fact]
        public void MatMulTransposeA_CpuVsCuda()
        {
            if (!CudaReady()) return;

            var bufferA = RandomBuffer(31 * 19, 5);
            var bufferB = RandomBuffer(31 * 17, 6);

            var cpu = TensorOps.MatMulTransposeA(CpuTensor(bufferA, 31, 19), CpuTensor(bufferB, 31, 17));
            var cuda = TensorOps.MatMulTransposeA(CudaTensor(bufferA, 31, 19), CudaTensor(bufferB, 31, 17));

            AssertBuffersClose(cpu.Buffer, cuda.Buffer, nameof(MatMulTransposeA_CpuVsCuda));
        }

        [Fact]
        public void SoftmaxRows_CpuVsCuda()
        {
            if (!CudaReady()) return;

            var buffer = RandomBuffer(17 * 23, 7);

            var cpu = TensorOps.SoftmaxRows(CpuTensor(buffer, 17, 23));
            var cuda = TensorOps.SoftmaxRows(CudaTensor(buffer, 17, 23));

            AssertBuffersClose(cpu.Buffer, cuda.Buffer, nameof(SoftmaxRows_CpuVsCuda));
        }

        [Fact]
        public void LayerNormRows_CpuVsCuda()
        {
            if (!CudaReady()) return;

            var buffer = RandomBuffer(13 * 11, 8);
            var gamma = RandomBuffer(11, 9);
            var beta = RandomBuffer(11, 10);

            var cpuResult = TensorOps.LayerNormRows(CpuTensor(buffer, 13, 11), CpuTensor(gamma, 11), CpuTensor(beta, 11), 1e-5f, out var cpuMean, out var cpuStd);
            var cudaResult = TensorOps.LayerNormRows(CudaTensor(buffer, 13, 11), CudaTensor(gamma, 11), CudaTensor(beta, 11), 1e-5f, out var cudaMean, out var cudaStd);

            AssertBuffersClose(cpuResult.Buffer, cudaResult.Buffer, nameof(LayerNormRows_CpuVsCuda));
            AssertBuffersClose(cpuMean, cudaMean, nameof(LayerNormRows_CpuVsCuda) + ".mean");
            AssertBuffersClose(cpuStd, cudaStd, nameof(LayerNormRows_CpuVsCuda) + ".std");
        }

        [Fact]
        public void MatMul_WithMismatchedDevices_Throws()
        {
            var a = new Tensor.Tensor(new[] { 2, 2 }, DeviceType.Cpu);
            var b = new Tensor.Tensor(new[] { 2, 2 }, DeviceType.Cuda);

            Assert.Throws<ArgumentException>(() => TensorOps.MatMul(a, b));
        }
    }
}
