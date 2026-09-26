using Neuraval.Core.Training.Stability;
using Xunit;

namespace Neuraval.Tests
{
    public class LossStabilityMonitorTests
    {
        [Fact]
        public void Observe_FirstObservation_SetsBestLossAndStepsObserved()
        {
            var monitor = new LossStabilityMonitor();

            var report = monitor.Observe(1.0f);

            Assert.Equal(1.0f, report.Loss);
            Assert.Equal(1.0f, report.BestLossSoFar);
            Assert.Equal(1, report.StepsObserved);
            Assert.False(report.IsDiverging);
        }

        [Fact]
        public void Observe_NaNLoss_ReportsHasNaN()
        {
            var monitor = new LossStabilityMonitor();

            var report = monitor.Observe(float.NaN);

            Assert.True(report.HasNaN);
            Assert.False(report.IsStable);
            Assert.False(report.IsDiverging);
        }

        [Fact]
        public void Observe_InfinityLoss_ReportsHasInfinity()
        {
            var monitor = new LossStabilityMonitor();

            var report = monitor.Observe(float.PositiveInfinity);

            Assert.True(report.HasInfinity);
            Assert.False(report.IsStable);
        }

        [Fact]
        public void Observe_DecreasingLosses_NeverDiverges()
        {
            var monitor = new LossStabilityMonitor(LossDivergenceOptions.Create(2.0f, 0));
            float[] losses = { 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };

            foreach (var loss in losses)
            {
                var report = monitor.Observe(loss);
                Assert.False(report.IsDiverging);
            }

            Assert.Equal(1.0f, monitor.BestLossSoFar);
        }

        [Fact]
        public void Observe_SpikeAfterWarmup_ReportsDiverging()
        {
            var monitor = new LossStabilityMonitor(LossDivergenceOptions.Create(2.0f, 2));

            monitor.Observe(1.0f);
            monitor.Observe(1.0f);
            var report = monitor.Observe(5.0f);

            Assert.True(report.IsDiverging);
            Assert.False(report.IsStable);
        }

        [Fact]
        public void Observe_SpikeDuringWarmup_DoesNotReportDiverging()
        {
            var monitor = new LossStabilityMonitor(LossDivergenceOptions.Create(2.0f, 5));

            monitor.Observe(1.0f);
            var report = monitor.Observe(100.0f);

            Assert.False(report.IsDiverging);
        }

        [Fact]
        public void Observe_BestLossSoFar_TracksMinimumAcrossSteps()
        {
            var monitor = new LossStabilityMonitor(LossDivergenceOptions.Create(10.0f, 0));

            monitor.Observe(3.0f);
            monitor.Observe(5.0f);
            monitor.Observe(1.0f);
            monitor.Observe(2.0f);

            Assert.Equal(1.0f, monitor.BestLossSoFar);
        }

        [Fact]
        public void Observe_IsStable_TrueForNormalLoss()
        {
            var monitor = new LossStabilityMonitor(LossDivergenceOptions.Create(2.0f, 0));

            var report = monitor.Observe(1.0f);

            Assert.True(report.IsStable);
        }

        [Fact]
        public void Observe_IsStable_FalseForDivergingLoss()
        {
            var monitor = new LossStabilityMonitor(LossDivergenceOptions.Create(2.0f, 0));

            monitor.Observe(1.0f);
            var report = monitor.Observe(5.0f);

            Assert.False(report.IsStable);
        }

        [Fact]
        public void Reset_ClearsState()
        {
            var monitor = new LossStabilityMonitor(LossDivergenceOptions.Create(2.0f, 0));

            monitor.Observe(1.0f);
            monitor.Observe(0.5f);
            monitor.Reset();

            Assert.Equal(float.PositiveInfinity, monitor.BestLossSoFar);
            Assert.Equal(0, monitor.StepsObserved);

            var report = monitor.Observe(10.0f);
            Assert.Equal(10.0f, report.BestLossSoFar);
            Assert.False(report.IsDiverging);
        }
    }
}
