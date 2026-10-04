namespace Neuraval.Cuda
{
    public sealed class CudaWeightCache : IDisposable
    {
        private readonly int _rows;
        private readonly int _cols;
        private CudaBuffer? _buffer;
        private bool _gpuDirty;

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
