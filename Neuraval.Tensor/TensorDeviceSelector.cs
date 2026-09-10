using Neuraval.Cuda;

namespace Neuraval.Tensor
{
    public static class TensorDeviceSelector
    {
        private static bool _probed;
        private static bool _gpuEnabled;

        public static DeviceType Current
        {
            get
            {
                if (!_probed)
                {
                    Probe();
                }

                return _gpuEnabled ? DeviceType.Cuda : DeviceType.Cpu;
            }
        }

        public static void ReportFailure()
        {
            _probed = true;
            _gpuEnabled = false;
        }

        public static void Reset()
        {
            _probed = false;
            _gpuEnabled = false;
        }

        private static void Probe()
        {
            _probed = true;

            try
            {
                if (!CudaDevice.IsAvailable())
                {
                    _gpuEnabled = false;
                    return;
                }

                if (!CudaDevice.Initialized)
                {
                    CudaDevice.Initialize();
                }

                _gpuEnabled = true;
            }
            catch (CudaException)
            {
                _gpuEnabled = false;
            }
        }
    }
}
