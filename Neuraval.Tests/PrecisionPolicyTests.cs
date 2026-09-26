using Neuraval.Core.Training.Precision;
using Xunit;

namespace Neuraval.Tests
{
    public class PrecisionPolicyTests
    {
        [Fact]
        public void Create_ExposesAllThreeDTypes()
        {
            var policy = PrecisionPolicy.Create(ComputeDType.Fp32, ComputeDType.Bf16, ComputeDType.Fp16);

            Assert.Equal(ComputeDType.Fp32, policy.ModelDType);
            Assert.Equal(ComputeDType.Bf16, policy.ComputeDType);
            Assert.Equal(ComputeDType.Fp16, policy.OptimizerDType);
        }

        [Fact]
        public void FullPrecision_AllDTypesAreFp32()
        {
            var policy = PrecisionPolicy.FullPrecision();

            Assert.Equal(ComputeDType.Fp32, policy.ModelDType);
            Assert.Equal(ComputeDType.Fp32, policy.ComputeDType);
            Assert.Equal(ComputeDType.Fp32, policy.OptimizerDType);
            Assert.False(policy.IsMixedPrecision);
        }

        [Fact]
        public void MixedFp16_KeepsModelAndOptimizerAtFp32()
        {
            var policy = PrecisionPolicy.MixedFp16();

            Assert.Equal(ComputeDType.Fp32, policy.ModelDType);
            Assert.Equal(ComputeDType.Fp16, policy.ComputeDType);
            Assert.Equal(ComputeDType.Fp32, policy.OptimizerDType);
            Assert.True(policy.IsMixedPrecision);
        }

        [Fact]
        public void MixedBf16_KeepsModelAndOptimizerAtFp32()
        {
            var policy = PrecisionPolicy.MixedBf16();

            Assert.Equal(ComputeDType.Fp32, policy.ModelDType);
            Assert.Equal(ComputeDType.Bf16, policy.ComputeDType);
            Assert.Equal(ComputeDType.Fp32, policy.OptimizerDType);
            Assert.True(policy.IsMixedPrecision);
        }

        [Fact]
        public void IsMixedPrecision_FalseWhenAllDTypesMatch()
        {
            var policy = PrecisionPolicy.Create(ComputeDType.Bf16, ComputeDType.Bf16, ComputeDType.Bf16);

            Assert.False(policy.IsMixedPrecision);
        }

        [Fact]
        public void IsMixedPrecision_TrueWhenComputeDTypeDiffers()
        {
            var policy = PrecisionPolicy.Create(ComputeDType.Fp32, ComputeDType.Fp16, ComputeDType.Fp32);

            Assert.True(policy.IsMixedPrecision);
        }

        [Fact]
        public void IsMixedPrecision_TrueWhenOptimizerDTypeDiffers()
        {
            var policy = PrecisionPolicy.Create(ComputeDType.Fp32, ComputeDType.Fp32, ComputeDType.Bf16);

            Assert.True(policy.IsMixedPrecision);
        }

        [Fact]
        public void IsMixedPrecision_TrueWhenModelDTypeDiffers()
        {
            var policy = PrecisionPolicy.Create(ComputeDType.Bf16, ComputeDType.Fp32, ComputeDType.Fp32);

            Assert.True(policy.IsMixedPrecision);
        }
    }
}
