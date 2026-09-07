using System.Runtime.InteropServices;
using Neuraval.Cuda.Native;

namespace Neuraval.Cuda
{
    public static class CudaDevice
    {
        private static bool _initialized;
        private static int _activeDeviceId = -1;

        public static bool Initialized => _initialized;

        public static int ActiveDeviceId => _activeDeviceId;

        public static bool IsAvailable()
        {
            try
            {
                return CudaNative.ncb_cuda_device_count() > 0;
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (EntryPointNotFoundException)
            {
                return false;
            }
        }

        public static int GetDeviceCount()
        {
            return CudaNative.ncb_cuda_device_count();
        }

        public static string GetDeviceName(int deviceId)
        {
            var buffer = new byte[256];
            int ok = CudaNative.ncb_cuda_get_device_name(deviceId, buffer, buffer.Length);

            if (ok == 0)
            {
                throw BuildException();
            }

            int nullIndex = Array.IndexOf(buffer, (byte)0);
            int length = nullIndex >= 0 ? nullIndex : buffer.Length;
            return System.Text.Encoding.ASCII.GetString(buffer, 0, length);
        }

        public static void Initialize(int deviceId = 0)
        {
            int ok = CudaNative.ncb_cuda_init(deviceId);

            if (ok == 0)
            {
                throw BuildException();
            }

            _initialized = true;
            _activeDeviceId = deviceId;
        }

        public static void Synchronize()
        {
            CudaNative.ncb_cuda_synchronize();
            ThrowIfError();
        }

        internal static void ThrowIfError()
        {
            int status = CudaNative.ncb_cuda_last_status();

            if (status != 0)
            {
                throw BuildException(status);
            }
        }

        private static CudaException BuildException()
        {
            int status = CudaNative.ncb_cuda_last_status();
            return BuildException(status);
        }

        private static CudaException BuildException(int status)
        {
            IntPtr messagePtr = CudaNative.ncb_cuda_status_message(status);
            string message = Marshal.PtrToStringAnsi(messagePtr) ?? "Error CUDA desconocido";
            return new CudaException(status, message);
        }
    }
}
