using System;
using Neuraval.Core.Evaluation;
using Xunit;

namespace Neuraval.Tests
{
    public class ReferenceEquivalenceCheckerTests
    {
        [Fact]
        public void Evaluate_AllMetricsWithinThresholds_IsEquivalent()
        {
            var report = new LogitsComparisonReport(0.00001f, 0.000001f, 0.0001f, 0.9999995f);
            var thresholds = ReferenceEquivalenceThresholds.Default();

            var verdict = ReferenceEquivalenceChecker.Evaluate(report, thresholds);

            Assert.True(verdict.IsEquivalent);
            Assert.Empty(verdict.FailureReasons);
        }

        [Fact]
        public void Evaluate_MaxAbsoluteErrorExceeds_ReportsReason()
        {
            var report = new LogitsComparisonReport(0.01f, 0.000001f, 0.0001f, 0.9999995f);
            var thresholds = ReferenceEquivalenceThresholds.Default();

            var verdict = ReferenceEquivalenceChecker.Evaluate(report, thresholds);

            Assert.False(verdict.IsEquivalent);
            Assert.Contains(verdict.FailureReasons, reason => reason.Contains("MaxAbsoluteError"));
        }

        [Fact]
        public void Evaluate_MeanAbsoluteErrorExceeds_ReportsReason()
        {
            var report = new LogitsComparisonReport(0.00001f, 0.01f, 0.0001f, 0.9999995f);
            var thresholds = ReferenceEquivalenceThresholds.Default();

            var verdict = ReferenceEquivalenceChecker.Evaluate(report, thresholds);

            Assert.False(verdict.IsEquivalent);
            Assert.Contains(verdict.FailureReasons, reason => reason.Contains("MeanAbsoluteError"));
        }

        [Fact]
        public void Evaluate_CosineSimilarityBelowMinimum_ReportsReason()
        {
            var report = new LogitsComparisonReport(0.00001f, 0.000001f, 0.0001f, 0.9f);
            var thresholds = ReferenceEquivalenceThresholds.Default();

            var verdict = ReferenceEquivalenceChecker.Evaluate(report, thresholds);

            Assert.False(verdict.IsEquivalent);
            Assert.Contains(verdict.FailureReasons, reason => reason.Contains("CosineSimilarity"));
        }

        [Fact]
        public void Evaluate_MultipleFailures_ReportsAllReasons()
        {
            var report = new LogitsComparisonReport(1f, 1f, 1f, 0f);
            var thresholds = ReferenceEquivalenceThresholds.Default();

            var verdict = ReferenceEquivalenceChecker.Evaluate(report, thresholds);

            Assert.False(verdict.IsEquivalent);
            Assert.Equal(3, verdict.FailureReasons.Count);
        }

        [Fact]
        public void Evaluate_NullReport_Throws()
        {
            Assert.Throws<ArgumentNullException>(() =>
                ReferenceEquivalenceChecker.Evaluate(null!, ReferenceEquivalenceThresholds.Default()));
        }

        [Fact]
        public void Evaluate_NullThresholds_Throws()
        {
            var report = new LogitsComparisonReport(0f, 0f, 0f, 1f);
            Assert.Throws<ArgumentNullException>(() => ReferenceEquivalenceChecker.Evaluate(report, null!));
        }
    }
}
