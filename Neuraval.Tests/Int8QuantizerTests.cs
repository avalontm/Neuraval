using System;
using Neuraval.Core.Quantization;
using Xunit;

namespace Neuraval.Tests
{
    public class Int8QuantizerTests
    {
        [Fact]
        public void QuantizeRowSymmetric_KnownRow_MatchesExpectedScaleAndData()
        {
            var flat = new float[] { 1f, -2f, 4f, 0f, 0f, 0f };

            var quantized = Int8Quantizer.QuantizeRowSymmetric(flat, 2, 3);

            Assert.Equal(4f / 127f, quantized.RowScales[0]);
            Assert.Equal(1f, quantized.RowScales[1]);
            Assert.Equal(0, quantized.Data[3]);
            Assert.Equal(0, quantized.Data[4]);
            Assert.Equal(0, quantized.Data[5]);
        }

        [Fact]
        public void QuantizeRowSymmetric_ZeroRow_DoesNotThrowAndScaleIsOne()
        {
            var flat = new float[] { 0f, 0f, 0f, 0f };

            var quantized = Int8Quantizer.QuantizeRowSymmetric(flat, 2, 2);

            Assert.All(quantized.RowScales, s => Assert.Equal(1f, s));
            Assert.All(quantized.Data, d => Assert.Equal(0, d));
        }

        [Fact]
        public void QuantizeRowSymmetric_SaturatesAtMaxLevel()
        {
            var flat = new float[] { 10f, -10f, 5f };

            var quantized = Int8Quantizer.QuantizeRowSymmetric(flat, 1, 3);

            Assert.Equal(127, quantized.Data[0]);
            Assert.Equal(-127, quantized.Data[1]);
        }

        [Fact]
        public void QuantizeRowSymmetric_2DOverload_MatchesFlatOverload()
        {
            var matrix = new float[,] { { 1f, 2f, 3f }, { -4f, -5f, -6f } };
            var flat = new float[] { 1f, 2f, 3f, -4f, -5f, -6f };

            var fromMatrix = Int8Quantizer.QuantizeRowSymmetric(matrix);
            var fromFlat = Int8Quantizer.QuantizeRowSymmetric(flat, 2, 3);

            Assert.Equal(fromFlat.Data, fromMatrix.Data);
            Assert.Equal(fromFlat.RowScales, fromMatrix.RowScales);
        }

        [Fact]
        public void QuantizeRowSymmetric_NullArray_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => Int8Quantizer.QuantizeRowSymmetric(null!, 1, 1));
        }

        [Fact]
        public void QuantizeRowSymmetric_MismatchedDimensions_Throws()
        {
            var flat = new float[] { 1f, 2f, 3f };

            Assert.Throws<ArgumentException>(() => Int8Quantizer.QuantizeRowSymmetric(flat, 2, 2));
        }

        [Fact]
        public void Dequantize_RoundTrip_IsCloseToOriginal()
        {
            var original = new float[] { 1f, -2f, 3.5f, -4.25f };

            var quantized = Int8Quantizer.QuantizeRowSymmetric(original, 1, 4);
            var dequantized = Int8Quantizer.Dequantize(quantized);

            for (int i = 0; i < original.Length; i++)
            {
                Assert.True(Math.Abs(original[i] - dequantized[i]) < 0.1f);
            }
        }

        [Fact]
        public void ComputeQuantizationError_IdenticalArrays_ReturnsZero()
        {
            var values = new float[] { 1f, 2f, 3f };

            var (maxAbsError, meanAbsError) = Int8Quantizer.ComputeQuantizationError(values, values);

            Assert.Equal(0f, maxAbsError);
            Assert.Equal(0f, meanAbsError);
        }

        [Fact]
        public void ComputeQuantizationError_KnownDifference_ComputesMaxAndMean()
        {
            var original = new float[] { 1f, 2f, 3f };
            var dequantized = new float[] { 1f, 3f, 3.5f };

            var (maxAbsError, meanAbsError) = Int8Quantizer.ComputeQuantizationError(original, dequantized);

            Assert.Equal(1f, maxAbsError);
            Assert.Equal(0.5f, meanAbsError, 3);
        }

        [Fact]
        public void ComputeQuantizationError_EmptyArrays_ReturnsZero()
        {
            var (maxAbsError, meanAbsError) = Int8Quantizer.ComputeQuantizationError(Array.Empty<float>(), Array.Empty<float>());

            Assert.Equal(0f, maxAbsError);
            Assert.Equal(0f, meanAbsError);
        }

        [Fact]
        public void ComputeQuantizationError_MismatchedLength_Throws()
        {
            Assert.Throws<ArgumentException>(() =>
                Int8Quantizer.ComputeQuantizationError(new float[] { 1f }, new float[] { 1f, 2f }));
        }

        [Fact]
        public void QuantizedMatrix_QuantizedByteSize_IsRowsTimesCols()
        {
            var quantized = Int8Quantizer.QuantizeRowSymmetric(new float[] { 1f, 2f, 3f, 4f }, 2, 2);

            Assert.Equal(4, quantized.QuantizedByteSize);
            Assert.Equal(16, quantized.OriginalByteSize);
        }
    }
}
