namespace Neuraval.Tensor
{
    public static class MixedPrecisionSettings
    {
        private static bool _enableFp16Inference = true;

        public static bool EnableFp16Inference => _enableFp16Inference;

        public static void Enable()
        {
            _enableFp16Inference = true;
        }

        public static void Disable()
        {
            _enableFp16Inference = false;
        }
    }
}
