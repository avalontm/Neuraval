using System;
using Neuraval.Core.Training.Precision;
using Xunit;

namespace Neuraval.Tests
{
    public class PrecisionConverterTests
    {
        [Fact]
        public void ToFloat16_ToFloat32_ExactForRepresentableValue()
        {
            var half = PrecisionConverter.ToFloat16(new[] { 1.5f });
            var roundTripped = PrecisionConverter.ToFloat32(half);

            Assert.Equal(1.5f, roundTripped[0]);
        }

        [Fact]
        public void ToFloat16_ToFloat32_LosesPrecisionForNonRepresentableValue()
        {
            var original = new[] { 3.14159265f };
            var roundTripped = PrecisionConverter.ToFloat32(PrecisionConverter.ToFloat16(original));

            float error = PrecisionConverter.MaxAbsoluteError(original, roundTripped);

            Assert.True(error > 0f);
            Assert.True(error < 0.01f);
        }

        [Fact]
        public void ToBFloat16_ToFloat32_ExactForRepresentableValue()
        {
            var bf16 = PrecisionConverter.ToBFloat16(new[] { 4.0f });
            var roundTripped = PrecisionConverter.ToFloat32(bf16);

            Assert.Equal(4.0f, roundTripped[0]);
        }

        [Fact]
        public void ToBFloat16_HasCoarserPrecisionThanFp16()
        {
            var original = new[] { 3.14159265f };

            float fp16Error = PrecisionConverter.MaxAbsoluteError(
                original, PrecisionConverter.ToFloat32(PrecisionConverter.ToFloat16(original)));

            float bf16Error = PrecisionConverter.MaxAbsoluteError(
                original, PrecisionConverter.ToFloat32(PrecisionConverter.ToBFloat16(original)));

            Assert.True(bf16Error >= fp16Error);
        }

        [Fact]
        public void RoundTrip_Fp32_ReturnsIndependentClone()
        {
            var original = new[] { 1.0f, 2.0f };
            var roundTripped = PrecisionConverter.RoundTrip(original, ComputeDType.Fp32);

            roundTripped[0] = 99f;

            Assert.Equal(1.0f, original[0]);
            Assert.Equal(2.0f, roundTripped[1]);
        }

        [Fact]
        public void RoundTrip_Fp16_MatchesExplicitConversion()
        {
            var original = new[] { 3.14159265f, -7.5f };

            var viaRoundTrip = PrecisionConverter.RoundTrip(original, ComputeDType.Fp16);
            var viaExplicit = PrecisionConverter.ToFloat32(PrecisionConverter.ToFloat16(original));

            Assert.Equal(viaExplicit, viaRoundTrip);
        }

        [Fact]
        public void RoundTrip_Bf16_MatchesExplicitConversion()
        {
            var original = new[] { 3.14159265f, -7.5f };

            var viaRoundTrip = PrecisionConverter.RoundTrip(original, ComputeDType.Bf16);
            var viaExplicit = PrecisionConverter.ToFloat32(PrecisionConverter.ToBFloat16(original));

            Assert.Equal(viaExplicit, viaRoundTrip);
        }

        [Fact]
        public void RoundTrip_UnknownDType_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => PrecisionConverter.RoundTrip(new[] { 1.0f }, (ComputeDType)99));
        }

        [Fact]
        public void MaxAbsoluteError_IdenticalArrays_ReturnsZero()
        {
            var error = PrecisionConverter.MaxAbsoluteError(new[] { 1.0f, 2.0f }, new[] { 1.0f, 2.0f });

            Assert.Equal(0f, error);
        }

        [Fact]
        public void MaxAbsoluteError_DifferentArrays_ReturnsMaxDifference()
        {
            var error = PrecisionConverter.MaxAbsoluteError(new[] { 1.0f, 2.0f, 5.0f }, new[] { 1.0f, 2.5f, 4.0f });

            Assert.Equal(1.0f, error, 5);
        }

        [Fact]
        public void MaxAbsoluteError_LengthMismatch_Throws()
        {
            Assert.Throws<ArgumentException>(() => PrecisionConverter.MaxAbsoluteError(new[] { 1.0f }, new[] { 1.0f, 2.0f }));
        }

        [Fact]
        public void ToFloat16_NullSource_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => PrecisionConverter.ToFloat16(null!));
        }

        [Fact]
        public void ToBFloat16_NullSource_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => PrecisionConverter.ToBFloat16(null!));
        }
    }
}
