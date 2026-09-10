using Neuraval.Cuda;

namespace Neuraval.Tensor.Backends.Cpu
{
    public sealed class CpuSimdBackend : ICpuTensorBackend
    {
        private readonly CpuScalarBackend _rowReductions = new CpuScalarBackend();

        public Tensor MatMul(Tensor a, Tensor b, int m, int k, int n)
        {
            var bTransposed = MatrixLayout.Transpose(b.Buffer, k, n);
            var result = new Tensor(new[] { m, n }, a.Device, a.DType);

            for (int i = 0; i < m; i++)
            {
                int aRowOffset = i * k;
                int resultRowOffset = i * n;

                for (int j = 0; j < n; j++)
                {
                    result.Buffer[resultRowOffset + j] = SimdKernels.DotProduct(a.Buffer, aRowOffset, bTransposed, j * k, k);
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
                    result.Buffer[resultRowOffset + j] = SimdKernels.DotProduct(a.Buffer, aRowOffset, b.Buffer, j * k, k) * scale;
                }
            }

            return result;
        }

        public Tensor MatMulTransposeA(Tensor a, Tensor b, int k, int m, int n)
        {
            var aTransposed = MatrixLayout.Transpose(a.Buffer, k, m);
            var bTransposed = MatrixLayout.Transpose(b.Buffer, k, n);
            var result = new Tensor(new[] { m, n }, a.Device, a.DType);

            for (int i = 0; i < m; i++)
            {
                int resultRowOffset = i * n;

                for (int j = 0; j < n; j++)
                {
                    result.Buffer[resultRowOffset + j] = SimdKernels.DotProduct(aTransposed, i * k, bTransposed, j * k, k);
                }
            }

            return result;
        }

        public Tensor MatMulCachedB(Tensor a, float[,] weights, CudaWeightCache weightCache, int m, int k, int n)
        {
            var weightsTransposed = weightCache.GetOrUploadCpuTransposed(weights);
            var result = new Tensor(new[] { m, n }, a.Device, a.DType);

            for (int i = 0; i < m; i++)
            {
                int aRowOffset = i * k;
                int resultRowOffset = i * n;

                for (int j = 0; j < n; j++)
                {
                    result.Buffer[resultRowOffset + j] = SimdKernels.DotProduct(a.Buffer, aRowOffset, weightsTransposed, j * k, k);
                }
            }

            return result;
        }

        public Tensor MatMulTransposeBCachedB(Tensor a, float[,] weights, CudaWeightCache weightCache, int m, int k, int n, float scale)
        {
            var flatWeights = weightCache.GetOrUploadCpuFlat(weights);
            var result = new Tensor(new[] { m, n }, a.Device, a.DType);

            for (int i = 0; i < m; i++)
            {
                int aRowOffset = i * k;
                int resultRowOffset = i * n;

                for (int j = 0; j < n; j++)
                {
                    result.Buffer[resultRowOffset + j] = SimdKernels.DotProduct(a.Buffer, aRowOffset, flatWeights, j * k, k) * scale;
                }
            }

            return result;
        }

        public Tensor Add(Tensor a, Tensor b)
        {
            var result = new Tensor((int[])a.Shape.Clone(), a.Device, a.DType);
            SimdKernels.Add(a.Buffer, b.Buffer, result.Buffer);
            return result;
        }

        public void AddInPlace(Tensor target, Tensor source, float scale)
        {
            SimdKernels.AddInPlace(target.Buffer, source.Buffer, scale);
        }

        public Tensor Scale(Tensor a, float scalar)
        {
            var result = new Tensor((int[])a.Shape.Clone(), a.Device, a.DType);
            SimdKernels.Scale(a.Buffer, scalar, result.Buffer);
            return result;
        }

        public Tensor Transpose(Tensor a, int rows, int cols)
        {
            var result = new Tensor(new[] { cols, rows }, a.Device, a.DType);
            var transposed = MatrixLayout.Transpose(a.Buffer, rows, cols);
            System.Array.Copy(transposed, result.Buffer, transposed.Length);
            return result;
        }

        public Tensor ReLU(Tensor a)
        {
            var result = new Tensor((int[])a.Shape.Clone(), a.Device, a.DType);
            SimdKernels.ReLU(a.Buffer, result.Buffer);
            return result;
        }

        public Tensor ReLUBackward(Tensor grad, Tensor activation)
        {
            var result = new Tensor((int[])grad.Shape.Clone(), grad.Device, grad.DType);
            SimdKernels.ReLUBackward(grad.Buffer, activation.Buffer, result.Buffer);
            return result;
        }

        public Tensor Multiply(Tensor a, Tensor b)
        {
            var result = new Tensor((int[])a.Shape.Clone(), a.Device, a.DType);
            SimdKernels.Multiply(a.Buffer, b.Buffer, result.Buffer);
            return result;
        }

        public Tensor SumRows(Tensor a, int rows, int cols)
        {
            return _rowReductions.SumRows(a, rows, cols);
        }

        public float Sum(Tensor a)
        {
            return SimdKernels.Sum(a.Buffer);
        }

        public Tensor SoftmaxRows(Tensor a, int rows, int cols)
        {
            return _rowReductions.SoftmaxRows(a, rows, cols);
        }

        public Tensor SoftmaxRowsBackward(Tensor gradOutput, Tensor softmaxOutput, int rows, int cols)
        {
            return _rowReductions.SoftmaxRowsBackward(gradOutput, softmaxOutput, rows, cols);
        }

        public Tensor LayerNormRows(Tensor a, Tensor gamma, Tensor beta, int rows, int cols, float epsilon, out float[] mean, out float[] std)
        {
            return _rowReductions.LayerNormRows(a, gamma, beta, rows, cols, epsilon, out mean, out std);
        }
    }
}
