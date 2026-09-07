namespace Neuraval.Cuda
{
    public sealed class CudaWeightCache : IDisposable
    {
        private readonly int _rows;
        private readonly int _cols;
        private CudaBuffer? _buffer;
        private bool _dirty;

        public CudaWeightCache(int rows, int cols)
        {
            _rows = rows;
            _cols = cols;
            _dirty = true;
        }

        public void Invalidate()
        {
            _dirty = true;
        }

        public CudaBuffer GetOrUpload(float[,] weights)
        {
            if (weights.GetLength(0) != _rows || weights.GetLength(1) != _cols)
            {
                throw new ArgumentException("Weight matrix shape does not match the cache");
            }

            if (_buffer == null)
            {
                _buffer = CudaBuffer.Allocate(_rows * _cols);
                _dirty = true;
            }

            if (_dirty)
            {
                var flat = new float[_rows * _cols];
                Buffer.BlockCopy(weights, 0, flat, 0, flat.Length * sizeof(float));
                _buffer.CopyFromHost(flat);
                _dirty = false;
            }

            return _buffer;
        }

        public void Dispose()
        {
            _buffer?.Dispose();
            _buffer = null;
        }
    }
}
