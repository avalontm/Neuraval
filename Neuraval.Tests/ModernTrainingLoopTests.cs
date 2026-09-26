using System;
using System.Collections.Generic;
using Neuraval.Core.Training.Loop;
using Xunit;

namespace Neuraval.Tests
{
    public class ModernTrainingLoopTests
    {
        [Fact]
        public void RunStep_IncrementsGlobalStepAndAccumulatesTokensSeen()
        {
            var loop = new ModernTrainingLoop(
                _ => new TrainingStepOutcome(1f, 0.5f, 100, 4, 1.0),
                step => 0.001f);

            loop.RunStep();
            loop.RunStep();

            Assert.Equal(2, loop.GlobalStep);
            Assert.Equal(200, loop.TokensSeen);
        }

        [Fact]
        public void RunStep_ComputesTokensPerSecondAndSamplesPerSecond()
        {
            var loop = new ModernTrainingLoop(
                _ => new TrainingStepOutcome(1f, 0.5f, 100, 4, 2.0),
                step => 0.001f);

            var metrics = loop.RunStep();

            Assert.Equal(50f, metrics.TokensPerSecond);
            Assert.Equal(2f, metrics.SamplesPerSecond);
        }

        [Fact]
        public void RunStep_PassesGlobalStepToLearningRateSchedule()
        {
            var observedSteps = new List<int>();

            var loop = new ModernTrainingLoop(
                _ => new TrainingStepOutcome(1f, 0.5f, 10, 1, 1.0),
                step =>
                {
                    observedSteps.Add(step);
                    return 0.001f;
                });

            loop.RunStep();
            loop.RunStep();

            Assert.Equal(new[] { 1, 2 }, observedSteps);
        }

        [Fact]
        public void RunStep_InvokesOnStepCompletedWithSameMetrics()
        {
            TrainingStepMetrics? observed = null;

            var loop = new ModernTrainingLoop(
                _ => new TrainingStepOutcome(1f, 0.5f, 10, 1, 1.0),
                step => 0.001f,
                onStepCompleted: metrics => observed = metrics);

            var returned = loop.RunStep();

            Assert.Same(returned, observed);
        }

        [Fact]
        public void RunStep_GpuMemorySamplerProvided_FlowsIntoMetrics()
        {
            var loop = new ModernTrainingLoop(
                _ => new TrainingStepOutcome(1f, 0.5f, 10, 1, 1.0),
                step => 0.001f,
                gpuMemorySampler: () => 2048L);

            var metrics = loop.RunStep();

            Assert.Equal(2048L, metrics.GpuMemoryBytes);
        }

        [Fact]
        public void RunStep_NoGpuMemorySampler_MetricsGpuMemoryIsNull()
        {
            var loop = new ModernTrainingLoop(
                _ => new TrainingStepOutcome(1f, 0.5f, 10, 1, 1.0),
                step => 0.001f);

            var metrics = loop.RunStep();

            Assert.Null(metrics.GpuMemoryBytes);
        }

        [Fact]
        public void Run_ExecutesRequestedNumberOfStepsInOrder()
        {
            var loop = new ModernTrainingLoop(
                _ => new TrainingStepOutcome(1f, 0.5f, 10, 1, 1.0),
                step => 0.001f);

            var results = loop.Run(3);

            Assert.Equal(3, results.Count);
            Assert.Equal(new[] { 1, 2, 3 }, new[] { results[0].GlobalStep, results[1].GlobalStep, results[2].GlobalStep });
            Assert.Equal(3, loop.History.Count);
        }

        [Fact]
        public void Constructor_NullExecuteStep_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => new ModernTrainingLoop(null!, step => 0.001f));
        }

        [Fact]
        public void Constructor_NullLearningRateSchedule_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => new ModernTrainingLoop(_ => new TrainingStepOutcome(1f, 0.5f, 10, 1, 1.0), null!));
        }

        [Fact]
        public void RunStep_ExecuteStepReturnsNull_Throws()
        {
            var loop = new ModernTrainingLoop(_ => null!, step => 0.001f);

            Assert.Throws<InvalidOperationException>(() => loop.RunStep());
        }

        [Fact]
        public void Run_NegativeSteps_Throws()
        {
            var loop = new ModernTrainingLoop(
                _ => new TrainingStepOutcome(1f, 0.5f, 10, 1, 1.0),
                step => 0.001f);

            Assert.Throws<ArgumentOutOfRangeException>(() => loop.Run(-1));
        }
    }
}
