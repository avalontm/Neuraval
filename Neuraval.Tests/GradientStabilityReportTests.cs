using Neuraval.Core.Training.Stability;
using Xunit;

namespace Neuraval.Tests
{
    public class GradientStabilityReportTests
    {
        [Fact]
        public void Create_ExposesAllFields()
        {
            var report = GradientStabilityReport.Create(5.0f, 2.5f, true, false, false);

            Assert.Equal(5.0f, report.NormBeforeClip);
            Assert.Equal(2.5f, report.NormAfterClip);
            Assert.True(report.WasClipped);
            Assert.False(report.HasNaN);
            Assert.False(report.HasInfinity);
        }

        [Fact]
        public void IsStable_TrueWhenNoNaNOrInfinity()
        {
            var report = GradientStabilityReport.Create(1.0f, 1.0f, false, false, false);

            Assert.True(report.IsStable);
        }

        [Fact]
        public void IsStable_FalseWhenHasNaN()
        {
            var report = GradientStabilityReport.Create(float.NaN, float.NaN, false, true, false);

            Assert.False(report.IsStable);
        }

        [Fact]
        public void IsStable_FalseWhenHasInfinity()
        {
            var report = GradientStabilityReport.Create(float.PositiveInfinity, float.PositiveInfinity, false, false, true);

            Assert.False(report.IsStable);
        }
    }
}
