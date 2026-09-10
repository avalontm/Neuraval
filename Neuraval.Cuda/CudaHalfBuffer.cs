using Neuraval.Cuda.Native;

namespace Neuraval.Cuda
{
    public sealed class CudaHalfBuffer : IDisposable
    {
        private const int BytesPerElement = 2;

        private IntPtr _devicePointer;
        private bool _disposed;

        public int Length { get; }

        public nuint SizeInBytes => (nuint)((long)Length * BytesPerElement);

        public IntPtr DevicePointer => _devicePointer;

        private CudaHalfBuffer(IntPtr devicePointer, int length)
        {
            _devicePointer = devicePointer;
            Length = length;
        }

        public static CudaHalfBuffer Allocate(int length)
        {
            nuint bytes = (nuint)((long)length * BytesPerElement);
            IntPtr ptr = CudaNative.ncb_cuda_alloc(bytes);

            if (ptr == IntPtr.Zero)
            {
                CudaDevice.ThrowIfError();
            }

            return new CudaHalfBuffer(ptr, length);
        }

        public static CudaHalfBuffer FromFloatBuffer(CudaBuffer source)
        {
            var half = Allocate(source.Length);
            CudaNative.ncb_cuda_cast_float_to_half_device(source.DevicePointer, half._devicePointer, source.Length);
            CudaDevice.ThrowIfError();
            return half;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            if (_devicePointer != IntPtr.Zero)
            {
                CudaNative.ncb_cuda_free(_devicePointer);
                _devicePointer = IntPtr.Zero;
            }

            _disposed = true;
            GC.SuppressFinalize(this);
        }

        ~CudaHalfBuffer()
        {
            Dispose();
        }
    }
}
