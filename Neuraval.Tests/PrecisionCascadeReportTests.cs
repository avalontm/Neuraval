using System;
using Neuraval.Core.Quantization;
using Xunit;

namespace Neuraval.Tests
{
    public class PrecisionCascadeReportTests
    {
        [Fact]
        public void Generate_KnownWeights_ReportsDimensions()
        {
            var weights = new float[] { 1f, -2f, 3f, -4f, 5f, -6f };

            var report = PrecisionCascadeReport.Generate(weights, 2, 3);

            Assert.Equal(2, report.Rows);
            Assert.Equal(3, report.Cols);
        }

        [Fact]
        public void Generate_Int8ErrorIsGreaterOrEqualThanFp16Error()
        {
            var random = new Random(42);
            var weights = new float[64];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = (float)(random.NextDouble() * 20 - 10);
            }

            var report = PrecisionCascadeReport.Generate(weights, 8, 8);

            Assert.True(report.Int8Error.MaxAbsError >= report.Fp16Error.MaxAbsError);
            Assert.True(report.Int8Error.MeanAbsError >= report.Fp16Error.MeanAbsError);
        }

        [Fact]
        public void Generate_ZeroWeights_ReportsZeroError()
        {
            var weights = new float[4];

            var report = PrecisionCascadeReport.Generate(weights, 2, 2);

            Assert.Equal(0f, report.Fp16Error.MaxAbsError);
            Assert.Equal(0f, report.Int8Error.MaxAbsError);
        }

        [Fact]
        public void Generate_ByteSizes_DecreaseAcrossStages()
        {
            var weights = new float[100];
            for (int i = 0; i < weights.Length; i++) weights[i] = i;

            var report = PrecisionCascadeReport.Generate(weights, 10, 10);

            Assert.Equal(400, report.Fp32ByteSize);
            Assert.Equal(200, report.Fp16ByteSize);
            Assert.Equal(100, report.Int8ByteSize);
            Assert.True(report.Fp32ByteSize > report.Fp16ByteSize);
            Assert.True(report.Fp16ByteSize > report.Int8ByteSize);
        }

        [Fact]
        public void Generate_NullWeights_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => PrecisionCascadeReport.Generate(null!, 1, 1));
        }

        [Fact]
        public void Generate_MismatchedDimensions_Throws()
        {
            var weights = new float[] { 1f, 2f, 3f };

            Assert.Throws<ArgumentException>(() => PrecisionCascadeReport.Generate(weights, 2, 2));
        }

        [Theory]
        [InlineData(0, 4)]
        [InlineData(4, 0)]
        [InlineData(-1, 4)]
        public void Generate_InvalidDimensions_Throws(int rows, int cols)
        {
            var weights = new float[4];

            Assert.Throws<ArgumentOutOfRangeException>(() => PrecisionCascadeReport.Generate(weights, rows, cols));
        }
    }
}
