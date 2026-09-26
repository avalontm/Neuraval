using Neuraval.Core.Training.Precision;
using Xunit;

namespace Neuraval.Tests
{
    public class BFloat16Tests
    {
        [Fact]
        public void FromSingle_PowerOfTwo_RoundTripsExactly()
        {
            var value = BFloat16.FromSingle(4.0f);

            Assert.Equal(4.0f, value.ToSingle());
        }

        [Fact]
        public void FromSingle_One_RoundTripsExactly()
        {
            var value = BFloat16.FromSingle(1.0f);

            Assert.Equal(1.0f, value.ToSingle());
        }

        [Fact]
        public void FromSingle_Zero_RoundTripsToZero()
        {
            var value = BFloat16.FromSingle(0.0f);

            Assert.Equal(0.0f, value.ToSingle());
        }

        [Fact]
        public void FromSingle_NegativeValue_PreservesSign()
        {
            var value = BFloat16.FromSingle(-2.0f);

            Assert.True(value.ToSingle() < 0f);
            Assert.Equal(-2.0f, value.ToSingle());
        }

        [Fact]
        public void FromSingle_NonRepresentableValue_LosesSomePrecisionButStaysClose()
        {
            float original = 3.14159265f;
            var value = BFloat16.FromSingle(original);
            float roundTripped = value.ToSingle();

            Assert.NotEqual(original, roundTripped);
            Assert.True(System.MathF.Abs(original - roundTripped) < 0.05f);
        }

        [Fact]
        public void FromSingle_NaN_RoundTripsToNaN()
        {
            var value = BFloat16.FromSingle(float.NaN);

            Assert.True(float.IsNaN(value.ToSingle()));
        }

        [Fact]
        public void FromSingle_PositiveInfinity_RoundTripsToInfinity()
        {
            var value = BFloat16.FromSingle(float.PositiveInfinity);

            Assert.True(float.IsPositiveInfinity(value.ToSingle()));
        }

        [Fact]
        public void ImplicitConversions_RoundTripThroughFloat()
        {
            BFloat16 value = 8.0f;
            float roundTripped = value;

            Assert.Equal(8.0f, roundTripped);
        }

        [Fact]
        public void Equals_SameValue_AreEqual()
        {
            var a = BFloat16.FromSingle(1.5f);
            var b = BFloat16.FromSingle(1.5f);

            Assert.True(a.Equals(b));
            Assert.Equal(a.GetHashCode(), b.GetHashCode());
        }

        [Fact]
        public void Equals_DifferentValue_AreNotEqual()
        {
            var a = BFloat16.FromSingle(1.5f);
            var b = BFloat16.FromSingle(2.5f);

            Assert.False(a.Equals(b));
        }
    }
}
