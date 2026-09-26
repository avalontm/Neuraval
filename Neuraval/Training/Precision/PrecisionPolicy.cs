namespace Neuraval.Core.Training.Precision
{
    public sealed class PrecisionPolicy
    {
        private PrecisionPolicy(ComputeDType modelDType, ComputeDType computeDType, ComputeDType optimizerDType)
        {
            ModelDType = modelDType;
            ComputeDType = computeDType;
            OptimizerDType = optimizerDType;
        }

        public ComputeDType ModelDType { get; }

        public ComputeDType ComputeDType { get; }

        public ComputeDType OptimizerDType { get; }

        public bool IsMixedPrecision =>
            ModelDType != ComputeDType || ModelDType != OptimizerDType || ComputeDType != OptimizerDType;

        public static PrecisionPolicy Create(ComputeDType modelDType, ComputeDType computeDType, ComputeDType optimizerDType)
        {
            return new PrecisionPolicy(modelDType, computeDType, optimizerDType);
        }

        public static PrecisionPolicy FullPrecision()
        {
            return new PrecisionPolicy(ComputeDType.Fp32, ComputeDType.Fp32, ComputeDType.Fp32);
        }

        public static PrecisionPolicy MixedFp16()
        {
            return new PrecisionPolicy(ComputeDType.Fp32, ComputeDType.Fp16, ComputeDType.Fp32);
        }

        public static PrecisionPolicy MixedBf16()
        {
            return new PrecisionPolicy(ComputeDType.Fp32, ComputeDType.Bf16, ComputeDType.Fp32);
        }
    }
}
