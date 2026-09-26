using System;
using Neuraval.Core.Evaluation;
using Xunit;

namespace Neuraval.Tests
{
    public class LogitsComparatorTests
    {
        [Fact]
        public void Compare_IdenticalVectors_AllErrorsZeroAndCosineOne()
        {
            var vector = new float[] { 1f, -2f, 3.5f, 0f };

            var report = LogitsComparator.Compare(vector, (float[])vector.Clone());

            Assert.Equal(0f, report.MaxAbsoluteError);
            Assert.Equal(0f, report.MeanAbsoluteError);
            Assert.Equal(0f, report.MeanRelativeError);
            Assert.Equal(1f, report.CosineSimilarity, 5);
        }

        [Fact]
        public void Compare_KnownDifference_MatchesHandComputedValues()
        {
            var reference = new float[] { 1f, 2f, 3f };
            var candidate = new float[] { 1f, 2f, 4f };

            var report = LogitsComparator.Compare(reference, candidate);

            Assert.Equal(1f, report.MaxAbsoluteError, 5);
            Assert.Equal(1f / 3f, report.MeanAbsoluteError, 5);

            float expectedCosine = (float)((1 * 1d + 2 * 2d + 3 * 4d) / (Math.Sqrt(1 + 4 + 9) * Math.Sqrt(1 + 4 + 16)));
            Assert.Equal(expectedCosine, report.CosineSimilarity, 5);
        }

        [Fact]
        public void Compare_BothVectorsAllZero_CosineSimilarityIsOne()
        {
            var reference = new float[] { 0f, 0f, 0f };
            var candidate = new float[] { 0f, 0f, 0f };

            var report = LogitsComparator.Compare(reference, candidate);

            Assert.Equal(1f, report.CosineSimilarity);
            Assert.Equal(0f, report.MaxAbsoluteError);
        }

        [Fact]
        public void Compare_OneVectorZeroOtherNonZero_CosineSimilarityIsZero()
        {
            var reference = new float[] { 0f, 0f };
            var candidate = new float[] { 1f, 1f };

            var report = LogitsComparator.Compare(reference, candidate);

            Assert.Equal(0f, report.CosineSimilarity);
        }

        [Fact]
        public void Compare_ReferenceNearZero_RelativeErrorUsesEpsilonFloor()
        {
            var reference = new float[] { 0f };
            var candidate = new float[] { 0.001f };

            var report = LogitsComparator.Compare(reference, candidate);

            Assert.True(float.IsFinite(report.MeanRelativeError));
            Assert.True(report.MeanRelativeError > 0f);
        }

        [Fact]
        public void Compare_NullReference_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => LogitsComparator.Compare(null!, new float[] { 1f }));
        }

        [Fact]
        public void Compare_NullCandidate_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => LogitsComparator.Compare(new float[] { 1f }, null!));
        }

        [Fact]
        public void Compare_LengthMismatch_Throws()
        {
            Assert.Throws<ArgumentException>(() => LogitsComparator.Compare(new float[] { 1f, 2f }, new float[] { 1f }));
        }

        [Fact]
        public void Compare_EmptyArrays_Throws()
        {
            Assert.Throws<ArgumentException>(() => LogitsComparator.Compare(Array.Empty<float>(), Array.Empty<float>()));
        }

        [Fact]
        public void Compare2D_MatchesFlattened1DResult()
        {
            var reference = new float[,] { { 1f, 2f }, { 3f, 4f } };
            var candidate = new float[,] { { 1f, 2f }, { 3f, 5f } };

            var report2D = LogitsComparator.Compare(reference, candidate);
            var report1D = LogitsComparator.Compare(new float[] { 1f, 2f, 3f, 4f }, new float[] { 1f, 2f, 3f, 5f });

            Assert.Equal(report1D.MaxAbsoluteError, report2D.MaxAbsoluteError);
            Assert.Equal(report1D.MeanAbsoluteError, report2D.MeanAbsoluteError);
            Assert.Equal(report1D.CosineSimilarity, report2D.CosineSimilarity);
        }

        [Fact]
        public void Compare2D_ShapeMismatch_Throws()
        {
            var reference = new float[,] { { 1f, 2f } };
            var candidate = new float[,] { { 1f, 2f }, { 3f, 4f } };

            Assert.Throws<ArgumentException>(() => LogitsComparator.Compare(reference, candidate));
        }

        [Fact]
        public void Compare2D_NullArguments_Throw()
        {
            var matrix = new float[,] { { 1f } };

            Assert.Throws<ArgumentNullException>(() => LogitsComparator.Compare((float[,])null!, matrix));
            Assert.Throws<ArgumentNullException>(() => LogitsComparator.Compare(matrix, (float[,])null!));
        }
    }
}
