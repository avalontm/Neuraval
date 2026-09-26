using System;
using Neuraval.Core.Quantization;
using Xunit;

namespace Neuraval.Tests
{
    public class Int8WeightCacheTests
    {
        [Fact]
        public void GetOrQuantize_FirstCall_QuantizesWeights()
        {
            var cache = new Int8WeightCache(2, 2);
            var weights = new float[,] { { 1f, 2f }, { 3f, 4f } };

            var quantized = cache.GetOrQuantize(weights);

            Assert.Equal(2, quantized.Rows);
            Assert.Equal(2, quantized.Cols);
        }

        [Fact]
        public void GetOrQuantize_SecondCallWithoutInvalidate_ReturnsCachedInstance()
        {
            var cache = new Int8WeightCache(2, 2);
            var weights = new float[,] { { 1f, 2f }, { 3f, 4f } };

            var first = cache.GetOrQuantize(weights);
            weights[0, 0] = 100f;
            var second = cache.GetOrQuantize(weights);

            Assert.Equal(first.Data, second.Data);
            Assert.Equal(first.RowScales, second.RowScales);
        }

        [Fact]
        public void GetOrQuantize_AfterInvalidate_Requantizes()
        {
            var cache = new Int8WeightCache(1, 2);
            var weights = new float[,] { { 1f, 2f } };

            var first = cache.GetOrQuantize(weights);
            weights[0, 1] = 200f;
            cache.Invalidate();
            var second = cache.GetOrQuantize(weights);

            Assert.NotEqual(first.RowScales[0], second.RowScales[0]);
        }

        [Fact]
        public void GetOrQuantize_MismatchedShape_Throws()
        {
            var cache = new Int8WeightCache(2, 2);
            var weights = new float[,] { { 1f, 2f, 3f } };

            Assert.Throws<ArgumentException>(() => cache.GetOrQuantize(weights));
        }
    }
}
