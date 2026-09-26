using Neuraval.Core.Quantization;
using Xunit;

namespace Neuraval.Tests
{
    public class Int8MatMulTests
    {
        [Fact]
        public void MatMulCachedB_MatchesFloatMatMul_WithinQuantizationTolerance()
        {
            var input = new Neuraval.Tensor.Tensor(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
            var weights = new float[,] { { 1f, 0f }, { 0f, 1f } };
            var cache = new Int8WeightCache(2, 2);

            var result = Int8MatMul.MatMulCachedB(input, weights, cache);

            Assert.Equal(new[] { 2, 2 }, result.Shape);
            Assert.Equal(1f, result.Buffer[0], 1);
            Assert.Equal(2f, result.Buffer[1], 1);
            Assert.Equal(3f, result.Buffer[2], 1);
            Assert.Equal(4f, result.Buffer[3], 1);
        }

        [Fact]
        public void MatMulCachedB_UsesCachedQuantization_AcrossCalls()
        {
            var input = new Neuraval.Tensor.Tensor(new float[] { 1f, 1f }, new[] { 1, 2 });
            var weights = new float[,] { { 2f, 0f }, { 0f, 2f } };
            var cache = new Int8WeightCache(2, 2);

            var first = Int8MatMul.MatMulCachedB(input, weights, cache);
            weights[0, 0] = 100f;
            var second = Int8MatMul.MatMulCachedB(input, weights, cache);

            Assert.Equal(first.Buffer, second.Buffer);
        }
    }
}
