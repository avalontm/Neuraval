using Neuraval.Cuda;

namespace Neuraval.Tensor.Backends.Cuda
{
    public sealed class CudaTensorBackend : ITensorBackend
    {
        public Tensor MatMul(Tensor a, Tensor b, int m, int k, int n)
        {
            var deviceResult = CudaMath.MatrixMultiply(a.Buffer, b.Buffer, m, k, n);
            return new Tensor(deviceResult, new[] { m, n }, a.Device, a.DType);
        }

        public Tensor MatMulTransposeB(Tensor a, Tensor b, int m, int k, int n, float scale)
        {
            var deviceResult = CudaMath.MatrixMultiplyTransposeB(a.Buffer, b.Buffer, m, k, n, scale);
            return new Tensor(deviceResult, new[] { m, n }, a.Device, a.DType);
        }

        public Tensor MatMulTransposeA(Tensor a, Tensor b, int k, int m, int n)
        {
            var deviceResult = CudaMath.MatrixMultiplyTransposeA(a.Buffer, b.Buffer, k, m, n);
            return new Tensor(deviceResult, new[] { m, n }, a.Device, a.DType);
        }

        public Tensor MatMulCachedB(Tensor a, float[,] weights, CudaWeightCache weightCache, int m, int k, int n)
        {
            var deviceResult = CudaMath.MatrixMultiplyCachedB(a.Buffer, weights, weightCache, m, k, n);
            return new Tensor(deviceResult, new[] { m, n }, a.Device, a.DType);
        }

        public Tensor MatMulTransposeBCachedB(Tensor a, float[,] weights, CudaWeightCache weightCache, int m, int k, int n, float scale)
        {
            var deviceResult = CudaMath.MatrixMultiplyTransposeBCachedB(a.Buffer, weights, weightCache, m, k, n, scale);
            return new Tensor(deviceResult, new[] { m, n }, a.Device, a.DType);
        }

        public Tensor SoftmaxRows(Tensor a, int rows, int cols)
        {
            var deviceResult = CudaMath.SoftmaxRows(a.Buffer, rows, cols);
            return new Tensor(deviceResult, new[] { rows, cols }, a.Device, a.DType);
        }

        public Tensor LayerNormRows(Tensor a, Tensor gamma, Tensor beta, int rows, int cols, float epsilon, out float[] mean, out float[] std)
        {
            var deviceResult = CudaMath.LayerNormRows(a.Buffer, gamma.Buffer, beta.Buffer, rows, cols, epsilon, out mean, out std);
            return new Tensor(deviceResult, new[] { rows, cols }, a.Device, a.DType);
        }
    }
}
