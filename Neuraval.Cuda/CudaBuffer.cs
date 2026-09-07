using Neuraval.Cuda.Native;

namespace Neuraval.Cuda
{
    public sealed class CudaBuffer : IDisposable
    {
        private IntPtr _devicePointer;
        private bool _disposed;

        public int Length { get; }

        public nuint SizeInBytes => (nuint)((long)Length * sizeof(float));

        public IntPtr DevicePointer => _devicePointer;

        private CudaBuffer(IntPtr devicePointer, int length)
        {
            _devicePointer = devicePointer;
            Length = length;
        }

        public static CudaBuffer Allocate(int length)
        {
            nuint bytes = (nuint)((long)length * sizeof(float));
            IntPtr ptr = CudaNative.ncb_cuda_alloc(bytes);

            if (ptr == IntPtr.Zero)
            {
                CudaDevice.ThrowIfError();
            }

            return new CudaBuffer(ptr, length);
        }

        public static CudaBuffer Upload(float[] data)
        {
            var buffer = Allocate(data.Length);
            buffer.CopyFromHost(data);
            return buffer;
        }

        public void CopyFromHost(float[] data)
        {
            CudaNative.ncb_cuda_copy_h2d(_devicePointer, data, SizeInBytes);
            CudaDevice.ThrowIfError();
        }

        public void CopyToHost(float[] destination)
        {
            CudaNative.ncb_cuda_copy_d2h(destination, _devicePointer, SizeInBytes);
            CudaDevice.ThrowIfError();
        }

        public float[] Download()
        {
            var result = new float[Length];
            CopyToHost(result);
            return result;
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

        ~CudaBuffer()
        {
            Dispose();
        }
    }
}
