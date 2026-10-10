using Neuraval.Cuda;
using Neuraval.Tensor;
using Neuraval.Tensor.Backends.Cpu;
using Xunit;
using TensorValue = Neuraval.Tensor.Tensor;

namespace Neuraval.Tests
{
    public sealed class CpuParallelBackendTests
    {
        [Fact]
        public void MatMulCachedB_ParallelizesSingleInputRowAcrossOutputTiles()
        {
            const int outputSize = 37;
            var input = new TensorValue(new[] { 1, 2 });
            input.Buffer[0] = 2f;
            input.Buffer[1] = 3f;

            var weights = new float[2, outputSize];
            for (int column = 0; column < outputSize; column++)
            {
                weights[0, column] = column + 1;
                weights[1, column] = (column + 1) * 10;
            }

            var backend = new CpuParallelBackend();
            backend.SetNumThreads(2);
            var output = backend.MatMulCachedB(input, weights, new CudaWeightCache(2, outputSize), 1, 2, outputSize);

            for (int column = 0; column < outputSize; column++)
                Assert.Equal(32f * (column + 1), output.Buffer[column]);
        }

        [Fact]
        public void MatMulTransposeBCachedB_ParallelizesSingleInputRowAcrossOutputTiles()
        {
            const int outputSize = 37;
            var input = new TensorValue(new[] { 1, 2 });
            input.Buffer[0] = 2f;
            input.Buffer[1] = 3f;

            var weights = new float[outputSize, 2];
            for (int row = 0; row < outputSize; row++)
            {
                weights[row, 0] = row + 1;
                weights[row, 1] = (row + 1) * 10;
            }

            var backend = new CpuParallelBackend();
            backend.SetNumThreads(2);
            var output = backend.MatMulTransposeBCachedB(input, weights, new CudaWeightCache(outputSize, 2), 1, 2, outputSize, 1f);

            for (int column = 0; column < outputSize; column++)
                Assert.Equal(32f * (column + 1), output.Buffer[column]);
        }

        [Fact]
        public void CpuTransposeCache_DoesNotAllocateASecondUntransposedCopy()
        {
            var cache = new CudaWeightCache(2, 3);
            cache.GetOrUploadCpuTransposed(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });

            var flatCache = typeof(CudaWeightCache).GetField("_cpuFlatDirect", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
            Assert.Null(flatCache.GetValue(cache));
        }
    }
}
