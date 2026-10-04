namespace Neuraval.Core.Quantization
{
    public static class Int8InferenceSettings
    {
        private static bool _enableInt8Cpu;

        public static bool EnableInt8Cpu => _enableInt8Cpu;

        public static void Enable()
        {
            _enableInt8Cpu = true;
        }

        public static void Disable()
        {
            _enableInt8Cpu = false;
        }
    }
}
