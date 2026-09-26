using System;
using Neuraval.Core.Training.Stability;
using Xunit;

namespace Neuraval.Tests
{
    public class GradientClipperTests
    {
        [Fact]
        public void ComputeGlobalNorm_SingleTensor_MatchesEuclideanNorm()
        {
            var norm = GradientClipper.ComputeGlobalNorm(new[] { new[] { 3f, 4f } });

            Assert.Equal(5f, norm, 5);
        }

        [Fact]
        public void ComputeGlobalNorm_MultipleTensors_CombinesAcrossAll()
        {
            var norm = GradientClipper.ComputeGlobalNorm(new[] { new[] { 3f, 4f }, new[] { 0f, 12f } });

            Assert.Equal(13f, norm, 5);
        }

        [Fact]
        public void ComputeGlobalNorm_NullList_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => GradientClipper.ComputeGlobalNorm(null!));
        }

        [Fact]
        public void ContainsNaN_DetectsNaN()
        {
            var hasNaN = GradientClipper.ContainsNaN(new[] { new[] { 1f, float.NaN } });

            Assert.True(hasNaN);
        }

        [Fact]
        public void ContainsNaN_NoNaN_ReturnsFalse()
        {
            var hasNaN = GradientClipper.ContainsNaN(new[] { new[] { 1f, 2f } });

            Assert.False(hasNaN);
        }

        [Fact]
        public void ContainsInfinity_DetectsInfinity()
        {
            var hasInfinity = GradientClipper.ContainsInfinity(new[] { new[] { 1f, float.PositiveInfinity } });

            Assert.True(hasInfinity);
        }

        [Fact]
        public void ContainsInfinity_NoInfinity_ReturnsFalse()
        {
            var hasInfinity = GradientClipper.ContainsInfinity(new[] { new[] { 1f, 2f } });

            Assert.False(hasInfinity);
        }

        [Fact]
        public void ClipByGlobalNorm_NormBelowMax_NoScalingApplied()
        {
            var tensor = new[] { 1f, 2f };
            var report = GradientClipper.ClipByGlobalNorm(new[] { tensor }, GradientClipOptions.Create(100f));

            Assert.False(report.WasClipped);
            Assert.Equal(1f, tensor[0], 5);
            Assert.Equal(2f, tensor[1], 5);
            Assert.Equal(report.NormBeforeClip, report.NormAfterClip, 5);
        }

        [Fact]
        public void ClipByGlobalNorm_NormAboveMax_ScalesTensorProportionally()
        {
            var tensor = new[] { 3f, 4f };
            var report = GradientClipper.ClipByGlobalNorm(new[] { tensor }, GradientClipOptions.Create(2.5f));

            Assert.True(report.WasClipped);
            Assert.Equal(5f, report.NormBeforeClip, 5);
            Assert.Equal(2.5f, report.NormAfterClip, 5);
            Assert.Equal(1.5f, tensor[0], 5);
            Assert.Equal(2.0f, tensor[1], 5);
        }

        [Fact]
        public void ClipByGlobalNorm_ScalesMultipleTensorsConsistently()
        {
            var tensorA = new[] { 3f, 4f };
            var tensorB = new[] { 0f, 12f };
            var report = GradientClipper.ClipByGlobalNorm(new[] { tensorA, tensorB }, GradientClipOptions.Create(6.5f));

            Assert.True(report.WasClipped);
            Assert.Equal(13f, report.NormBeforeClip, 5);
            Assert.Equal(1.5f, tensorA[0], 5);
            Assert.Equal(2.0f, tensorA[1], 5);
            Assert.Equal(0.0f, tensorB[0], 5);
            Assert.Equal(6.0f, tensorB[1], 5);
        }

        [Fact]
        public void ClipByGlobalNorm_WithNaN_ReportsUnstableAndDoesNotScale()
        {
            var tensor = new[] { 1f, float.NaN };
            var report = GradientClipper.ClipByGlobalNorm(new[] { tensor }, GradientClipOptions.Create(1f));

            Assert.True(report.HasNaN);
            Assert.False(report.WasClipped);
            Assert.False(report.IsStable);
            Assert.Equal(1f, tensor[0], 5);
            Assert.True(float.IsNaN(tensor[1]));
        }

        [Fact]
        public void ClipByGlobalNorm_WithInfinity_ReportsUnstableAndDoesNotScale()
        {
            var tensor = new[] { 1f, float.PositiveInfinity };
            var report = GradientClipper.ClipByGlobalNorm(new[] { tensor }, GradientClipOptions.Create(1f));

            Assert.True(report.HasInfinity);
            Assert.False(report.WasClipped);
            Assert.False(report.IsStable);
            Assert.Equal(1f, tensor[0], 5);
            Assert.True(float.IsPositiveInfinity(tensor[1]));
        }

        [Fact]
        public void ClipByGlobalNorm_ZeroNorm_NoScalingApplied()
        {
            var tensor = new[] { 0f, 0f };
            var report = GradientClipper.ClipByGlobalNorm(new[] { tensor }, GradientClipOptions.Create(1f));

            Assert.False(report.WasClipped);
            Assert.Equal(0f, report.NormBeforeClip, 5);
            Assert.Equal(0f, tensor[0], 5);
            Assert.Equal(0f, tensor[1], 5);
        }

        [Fact]
        public void ClipByGlobalNorm_NullGradientTensors_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => GradientClipper.ClipByGlobalNorm(null!, GradientClipOptions.Create(1f)));
        }

        [Fact]
        public void ClipByGlobalNorm_NullOptions_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => GradientClipper.ClipByGlobalNorm(new[] { new[] { 1f } }, null!));
        }
    }
}
