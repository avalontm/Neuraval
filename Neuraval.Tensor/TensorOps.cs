using System;
using Neuraval.Cuda;
using Neuraval.Tensor.Backends;
using Neuraval.Tensor.Backends.Cpu;
using Neuraval.Tensor.Backends.Cuda;

namespace Neuraval.Tensor
{
    public static class TensorOps
    {
        private static readonly CudaTensorBackend CudaBackend = new CudaTensorBackend();

        public static void SetNumThreads(int numThreads)
        {
            CpuBackendSelector.SetNumThreads(numThreads);
        }

        public static int GetNumThreads()
        {
            return CpuBackendSelector.GetNumThreads();
        }

        private static void EnsureRank(Tensor tensor, int expectedRank, string paramName)
        {
            if (tensor.Rank != expectedRank)
            {
                throw new ArgumentException($"Se esperaba un tensor de rank {expectedRank} pero {paramName} tiene rank {tensor.Rank}");
            }
        }

        private static void EnsureSameShape(Tensor a, Tensor b)
        {
            if (!a.ShapeEquals(b))
            {
                throw new ArgumentException("Los tensores deben tener el mismo shape");
            }
        }

        private static void EnsureSameDevice(Tensor a, Tensor b)
        {
            if (a.Device != b.Device)
            {
                throw new ArgumentException("Los tensores deben estar en el mismo device");
            }
        }

        private static ITensorBackend SelectDeviceAwareBackend(Tensor a, long workload)
        {
            return a.Device == DeviceType.Cuda ? CudaBackend : CpuBackendSelector.SelectFor(workload);
        }

        private static ICpuTensorBackend SelectCpuBackend(long workload)
        {
            return CpuBackendSelector.SelectFor(workload);
        }

        public static Tensor MatMul(Tensor a, Tensor b)
        {
            EnsureRank(a, 2, nameof(a));
            EnsureRank(b, 2, nameof(b));
            EnsureSameDevice(a, b);

            int m = a.Shape[0];
            int k = a.Shape[1];
            int k2 = b.Shape[0];
            int n = b.Shape[1];

            if (k != k2)
            {
                throw new ArgumentException($"Dimensiones incompatibles para MatMul: {m}x{k} · {k2}x{n}");
            }

            var backend = SelectDeviceAwareBackend(a, (long)m * k * n);
            return backend.MatMul(a, b, m, k, n);
        }

        public static Tensor MatMulTransposeB(Tensor a, Tensor b, float scale = 1.0f)
        {
            EnsureRank(a, 2, nameof(a));
            EnsureRank(b, 2, nameof(b));
            EnsureSameDevice(a, b);

            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = b.Shape[0];
            int k2 = b.Shape[1];

            if (k != k2)
            {
                throw new ArgumentException($"Dimensiones incompatibles para MatMulTransposeB: {m}x{k} · ({n}x{k2})^T");
            }

            var backend = SelectDeviceAwareBackend(a, (long)m * k * n);
            return backend.MatMulTransposeB(a, b, m, k, n, scale);
        }

        public static Tensor MatMulTransposeA(Tensor a, Tensor b)
        {
            EnsureRank(a, 2, nameof(a));
            EnsureRank(b, 2, nameof(b));
            EnsureSameDevice(a, b);

            int k = a.Shape[0];
            int m = a.Shape[1];
            int k2 = b.Shape[0];
            int n = b.Shape[1];

            if (k != k2)
            {
                throw new ArgumentException($"Dimensiones incompatibles para MatMulTransposeA: ({k}x{m})^T · {k2}x{n}");
            }

            var backend = SelectDeviceAwareBackend(a, (long)m * k * n);
            return backend.MatMulTransposeA(a, b, k, m, n);
        }

        public static Tensor MatMulCachedB(Tensor a, float[,] weights, CudaWeightCache weightCache)
        {
            EnsureRank(a, 2, nameof(a));

            int m = a.Shape[0];
            int k = a.Shape[1];
            int wRows = weights.GetLength(0);
            int n = weights.GetLength(1);

            if (wRows != k)
            {
                throw new ArgumentException($"Dimensiones incompatibles para MatMulCachedB: {m}x{k} · {wRows}x{n}");
            }

            var backend = SelectDeviceAwareBackend(a, (long)m * k * n);
            return backend.MatMulCachedB(a, weights, weightCache, m, k, n);
        }

        public static Tensor MatMulTransposeBCachedB(Tensor a, float[,] weights, CudaWeightCache weightCache, float scale = 1.0f)
        {
            EnsureRank(a, 2, nameof(a));

            int m = a.Shape[0];
            int k = a.Shape[1];
            int n = weights.GetLength(0);
            int wCols = weights.GetLength(1);

            if (wCols != k)
            {
                throw new ArgumentException($"Dimensiones incompatibles para MatMulTransposeBCachedB: {m}x{k} · ({n}x{wCols})^T");
            }

            var backend = SelectDeviceAwareBackend(a, (long)m * k * n);
            return backend.MatMulTransposeBCachedB(a, weights, weightCache, m, k, n, scale);
        }

        public static Tensor Add(Tensor a, Tensor b)
        {
            EnsureSameShape(a, b);

            var backend = SelectCpuBackend(a.Buffer.Length);
            return backend.Add(a, b);
        }

        public static void AddInPlace(Tensor target, Tensor source, float scale = 1.0f)
        {
            EnsureSameShape(target, source);

            var backend = SelectCpuBackend(target.Buffer.Length);
            backend.AddInPlace(target, source, scale);
        }

        public static Tensor Scale(Tensor a, float scalar)
        {
            var backend = SelectCpuBackend(a.Buffer.Length);
            return backend.Scale(a, scalar);
        }

        public static void Clear(Tensor a)
        {
            Array.Clear(a.Buffer, 0, a.Buffer.Length);
        }

        public static Tensor SliceColumns(Tensor a, int startCol, int endCol)
        {
            EnsureRank(a, 2, nameof(a));

            int rows = a.Shape[0];
            int cols = a.Shape[1];
            int numCols = endCol - startCol;

            if (startCol < 0 || endCol > cols || numCols <= 0)
            {
                throw new ArgumentException($"Rango de columnas inválido [{startCol}, {endCol}) para un tensor de {cols} columnas");
            }

            var result = new Tensor(new[] { rows, numCols }, a.Device, a.DType);

            // Cada fila es un bloque contiguo en memoria (row-major); el rango de
            // columnas pedido también es contiguo dentro de esa fila, así que
            // alcanza con un memcpy por fila en vez de una copia elemento a elemento.
            for (int i = 0; i < rows; i++)
            {
                Buffer.BlockCopy(a.Buffer, (i * cols + startCol) * sizeof(float), result.Buffer, i * numCols * sizeof(float), numCols * sizeof(float));
            }

            return result;
        }

        public static void SetColumns(Tensor target, Tensor source, int startCol)
        {
            EnsureRank(target, 2, nameof(target));
            EnsureRank(source, 2, nameof(source));

            int rows = target.Shape[0];
            int cols = target.Shape[1];
            int sourceCols = source.Shape[1];

            if (source.Shape[0] != rows || startCol < 0 || startCol + sourceCols > cols)
            {
                throw new ArgumentException("El tensor origen no encaja en el destino a partir de startCol");
            }

            for (int i = 0; i < rows; i++)
            {
                Buffer.BlockCopy(source.Buffer, i * sourceCols * sizeof(float), target.Buffer, (i * cols + startCol) * sizeof(float), sourceCols * sizeof(float));
            }
        }

        public static Tensor GetBatchSlice(Tensor batch, int batchIndex)
        {
            EnsureRank(batch, 3, nameof(batch));

            int rows = batch.Shape[1];
            int cols = batch.Shape[2];
            int sliceSize = rows * cols;

            if (batchIndex < 0 || batchIndex >= batch.Shape[0])
            {
                throw new ArgumentOutOfRangeException(nameof(batchIndex));
            }

            var result = new Tensor(new[] { rows, cols }, batch.Device, batch.DType);

            // batch[batchIndex, *, *] es un bloque contiguo (batch es la dimensión
            // más externa en el layout row-major), así que es un único memcpy.
            Buffer.BlockCopy(batch.Buffer, batchIndex * sliceSize * sizeof(float), result.Buffer, 0, sliceSize * sizeof(float));

            return result;
        }

        public static void SetBatchSlice(Tensor batch, int batchIndex, Tensor slice)
        {
            EnsureRank(batch, 3, nameof(batch));
            EnsureRank(slice, 2, nameof(slice));

            int rows = batch.Shape[1];
            int cols = batch.Shape[2];
            int sliceSize = rows * cols;

            if (slice.Shape[0] != rows || slice.Shape[1] != cols)
            {
                throw new ArgumentException("El shape del slice no coincide con [rows, cols] del batch");
            }

            if (batchIndex < 0 || batchIndex >= batch.Shape[0])
            {
                throw new ArgumentOutOfRangeException(nameof(batchIndex));
            }

            Buffer.BlockCopy(slice.Buffer, 0, batch.Buffer, batchIndex * sliceSize * sizeof(float), sliceSize * sizeof(float));
        }

        public static Tensor Transpose(Tensor a)
        {
            EnsureRank(a, 2, nameof(a));

            int rows = a.Shape[0];
            int cols = a.Shape[1];

            var backend = SelectCpuBackend((long)rows * cols);
            return backend.Transpose(a, rows, cols);
        }

        public static Tensor ReLU(Tensor a)
        {
            var backend = SelectCpuBackend(a.Buffer.Length);
            return backend.ReLU(a);
        }

        public static Tensor ReLUBackward(Tensor grad, Tensor activation)
        {
            EnsureSameShape(grad, activation);

            var backend = SelectCpuBackend(grad.Buffer.Length);
            return backend.ReLUBackward(grad, activation);
        }

        public static Tensor Multiply(Tensor a, Tensor b)
        {
            EnsureSameShape(a, b);

            var backend = SelectCpuBackend(a.Buffer.Length);
            return backend.Multiply(a, b);
        }

        public static Tensor SumRows(Tensor a)
        {
            EnsureRank(a, 2, nameof(a));

            int rows = a.Shape[0];
            int cols = a.Shape[1];

            var backend = SelectCpuBackend((long)rows * cols);
            return backend.SumRows(a, rows, cols);
        }

        public static float Sum(Tensor a)
        {
            var backend = SelectCpuBackend(a.Buffer.Length);
            return backend.Sum(a);
        }

        public static Tensor SoftmaxRows(Tensor a)
        {
            EnsureRank(a, 2, nameof(a));

            int rows = a.Shape[0];
            int cols = a.Shape[1];

            var backend = SelectDeviceAwareBackend(a, (long)rows * cols);
            return backend.SoftmaxRows(a, rows, cols);
        }

        public static Tensor SoftmaxRowsBackward(Tensor gradOutput, Tensor softmaxOutput)
        {
            EnsureRank(gradOutput, 2, nameof(gradOutput));
            EnsureRank(softmaxOutput, 2, nameof(softmaxOutput));
            EnsureSameShape(gradOutput, softmaxOutput);

            int rows = gradOutput.Shape[0];
            int cols = gradOutput.Shape[1];

            var backend = SelectCpuBackend((long)rows * cols);
            return backend.SoftmaxRowsBackward(gradOutput, softmaxOutput, rows, cols);
        }

        public static Tensor LayerNormRows(Tensor a, Tensor gamma, Tensor beta, float epsilon, out float[] mean, out float[] std)
        {
            EnsureRank(a, 2, nameof(a));
            EnsureRank(gamma, 1, nameof(gamma));
            EnsureRank(beta, 1, nameof(beta));
            EnsureSameDevice(a, gamma);
            EnsureSameDevice(a, beta);

            int rows = a.Shape[0];
            int cols = a.Shape[1];

            if (gamma.Shape[0] != cols || beta.Shape[0] != cols)
            {
                throw new ArgumentException("gamma y beta deben tener la misma dimensión que las columnas de a");
            }

            var backend = SelectDeviceAwareBackend(a, (long)rows * cols);
            return backend.LayerNormRows(a, gamma, beta, rows, cols, epsilon, out mean, out std);
        }
    }
}