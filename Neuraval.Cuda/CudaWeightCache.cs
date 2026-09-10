namespace Neuraval.Cuda
{
    public sealed class CudaWeightCache : IDisposable
    {
        private readonly int _rows;
        private readonly int _cols;
        private CudaBuffer? _buffer;
        private bool _gpuDirty;

        // Cache CPU-side, independiente del buffer de GPU: los backends
        // CpuParallelBackend/CpuSimdBackend necesitan la matriz de pesos
        // aplanada (para MatMulTransposeBCachedB) y transpuesta (para
        // MatMulCachedB) para que el dot product vectorizado tenga acceso
        // contiguo. Antes de esto, ambos backends recalculaban flatten+transpose
        // desde cero en cada Forward/Backward, ignorando por completo este
        // cache (que hasta ahora solo servía al backend CUDA). Comparte la
        // matriz de origen con el path de GPU, pero nunca toca CudaBuffer ni
        // asigna memoria de GPU, así que es seguro de usar en ejecuciones sin
        // CUDA disponible.
        private float[]? _cpuFlatDirect;
        private bool _cpuFlatDirty;
        private float[]? _cpuFlatTransposed;
        private bool _cpuTransposedDirty;

        public CudaWeightCache(int rows, int cols)
        {
            _rows = rows;
            _cols = cols;
            _gpuDirty = true;
            _cpuFlatDirty = true;
            _cpuTransposedDirty = true;
        }

        public void Invalidate()
        {
            _gpuDirty = true;
            _cpuFlatDirty = true;
            _cpuTransposedDirty = true;
        }

        public CudaBuffer GetOrUpload(float[,] weights)
        {
            EnsureShape(weights);

            if (_buffer == null)
            {
                _buffer = CudaBuffer.Allocate(_rows * _cols);
                _gpuDirty = true;
            }

            if (_gpuDirty)
            {
                _buffer.CopyFromHost(GetOrUploadCpuFlat(weights));
                _gpuDirty = false;
            }

            return _buffer;
        }

        /// <summary>
        /// Versión aplanada (row-major, sin transponer) de <paramref name="weights"/>,
        /// recalculada solo cuando el cache está sucio. Usada por
        /// MatMulTransposeBCachedB en los backends CPU, donde se necesita
        /// weights[j, p] directo.
        /// </summary>
        public float[] GetOrUploadCpuFlat(float[,] weights)
        {
            EnsureShape(weights);

            if (_cpuFlatDirect == null || _cpuFlatDirty)
            {
                var flat = new float[_rows * _cols];
                Buffer.BlockCopy(weights, 0, flat, 0, flat.Length * sizeof(float));
                _cpuFlatDirect = flat;
                _cpuFlatDirty = false;
            }

            return _cpuFlatDirect;
        }

        /// <summary>
        /// Versión aplanada Y transpuesta de <paramref name="weights"/> (shape
        /// lógico cols x rows), recalculada solo cuando el cache está sucio.
        /// Usada por MatMulCachedB en los backends CPU, donde conviene tener
        /// las columnas de weights contiguas para el dot product vectorizado.
        /// </summary>
        public float[] GetOrUploadCpuTransposed(float[,] weights)
        {
            EnsureShape(weights);

            if (_cpuFlatTransposed == null || _cpuTransposedDirty)
            {
                var flat = GetOrUploadCpuFlat(weights);
                var transposed = new float[_rows * _cols];

                for (int i = 0; i < _rows; i++)
                {
                    int rowOffset = i * _cols;

                    for (int j = 0; j < _cols; j++)
                    {
                        transposed[j * _rows + i] = flat[rowOffset + j];
                    }
                }

                _cpuFlatTransposed = transposed;
                _cpuTransposedDirty = false;
            }

            return _cpuFlatTransposed;
        }

        private void EnsureShape(float[,] weights)
        {
            if (weights.GetLength(0) != _rows || weights.GetLength(1) != _cols)
            {
                throw new ArgumentException("Weight matrix shape does not match the cache");
            }
        }

        public void Dispose()
        {
            _buffer?.Dispose();
            _buffer = null;
        }
    }
}
