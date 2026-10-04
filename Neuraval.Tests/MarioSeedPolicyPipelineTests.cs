using System;
using System.Linq;
using Neuraval.Evolution.MarioBridge;
using Neuraval.Evolution.Neat;
using Xunit;

namespace Neuraval.Tests
{
    public class MarioSeedPolicyPipelineTests
    {
        private static float[] BuildSampleInput(int length, int seed)
        {
            var random = new Random(seed);
            var input = new float[length];
            for (var i = 0; i < length; i++)
            {
                input[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }

            return input;
        }

        [Fact]
        public void NeatGenome_OutputNodes_UseSigmoid_LikeMarioPolicyNetwork()
        {
            var tracker = new NeatInnovationTracker(MarioAgent.InputCount + MarioAgent.OutputCount + 1);
            var random = new Random(7);
            var genome = NeatGenome.CreateInitial(MarioAgent.InputCount, MarioAgent.OutputCount, random, tracker);

            var input = BuildSampleInput(MarioAgent.InputCount, seed: 99);
            var output = genome.Evaluate(input);

            Assert.Equal(MarioAgent.OutputCount, output.Length);
            foreach (var value in output)
            {
                Assert.InRange(value, 0f, 1f);
            }
        }

        [Fact]
        public void AsGenome_ThenEvaluate_ReproducesMarioPolicyNetworkForward()
        {
            const int hiddenSize = 10;
            var random = new Random(2024);
            var policy = MarioPolicyNetwork.Create(MarioAgent.InputCount, MarioAgent.OutputCount, random, hiddenSize);

            var tracker = new NeatInnovationTracker(MarioAgent.InputCount + MarioAgent.OutputCount + 1);
            var genome = policy.AsGenome(tracker);

            for (var sample = 0; sample < 5; sample++)
            {
                var input = BuildSampleInput(MarioAgent.InputCount, seed: 1000 + sample);

                var expected = policy.Forward(input);
                var actual = genome.Evaluate(input);

                Assert.Equal(expected.Length, actual.Length);
                for (var i = 0; i < expected.Length; i++)
                {
                    Assert.True(
                        Math.Abs(expected[i] - actual[i]) < 1e-4f,
                        $"Output {i} difiere: politica={expected[i]:F6} genoma={actual[i]:F6} (input de prueba #{sample}).");
                }

                Assert.Equal(MarioAgentOutput.ToAction(expected), MarioAgentOutput.ToAction(actual));
            }
        }

        [Fact]
        public void SeededGenome_SurvivesOneNeatGeneration_AndKeepsShapeAndSigmoidOutputs()
        {
            const int populationSize = 12;
            const int hiddenSize = 6;

            var random = new Random(555);
            var tracker = new NeatInnovationTracker(MarioAgent.InputCount + MarioAgent.OutputCount + 1);

            var policy = MarioPolicyNetwork.Create(MarioAgent.InputCount, MarioAgent.OutputCount, random, hiddenSize);
            var seedGenome = policy.AsGenome(tracker);

            var maxNodeId = seedGenome.Nodes.Max(node => node.Id);
            var maxInnovation = seedGenome.Connections.Count == 0
                ? -1
                : seedGenome.Connections.Max(connection => connection.Innovation);
            tracker.FastForwardTo(maxNodeId + 1, maxInnovation + 1);

            var agents = new MarioAgent[populationSize];
            agents[0] = new MarioAgent(seedGenome.Clone());
            for (var i = 1; i < populationSize; i++)
            {
                agents[i] = MarioAgent.CreateRandom(random, tracker);
            }

            var options = new NeatEvolutionOptions();
            var strategy = new NeatEvolutionStrategy<MarioAgent>(random, tracker, options, genome => new MarioAgent(genome));

            var fitness = Enumerable.Range(0, populationSize)
                .Select(i => i == 0 ? 1000f : (float)random.Next(0, 50))
                .ToList();

            var nextGeneration = strategy.NextGeneration(agents, fitness);

            Assert.Equal(populationSize, nextGeneration.Count);

            var probeInput = BuildSampleInput(MarioAgent.InputCount, seed: 4242);

            foreach (var agent in nextGeneration)
            {
                Assert.Equal(MarioAgent.InputCount, agent.Genome.InputCount);
                Assert.Equal(MarioAgent.OutputCount, agent.Genome.OutputCount);

                var output = agent.Genome.Evaluate(probeInput);
                Assert.Equal(MarioAgent.OutputCount, output.Length);
                foreach (var value in output)
                {
                    Assert.InRange(value, 0f, 1f);
                }

                _ = MarioAgentOutput.ToAction(output);
            }
        }

        [Fact]
        public void SavedPolicy_RoundTrip_ThenSeeded_MatchesForwardBeforeSaving()
        {
            const int hiddenSize = 8;
            var policyPath = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(), $"mario_seed_policy_{Guid.NewGuid():N}.navm");

            try
            {
                var random = new Random(31337);
                var trained = MarioPolicyNetwork.Create(MarioAgent.InputCount, MarioAgent.OutputCount, random, hiddenSize);
                trained.Save(policyPath);

                var reloaded = MarioPolicyNetwork.Load(policyPath);
                Assert.NotNull(reloaded);

                var tracker = new NeatInnovationTracker(MarioAgent.InputCount + MarioAgent.OutputCount + 1);
                var genome = reloaded!.AsGenome(tracker);

                var input = BuildSampleInput(MarioAgent.InputCount, seed: 8);
                var expected = trained.Forward(input);
                var actual = genome.Evaluate(input);

                for (var i = 0; i < expected.Length; i++)
                {
                    Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-4f, $"Output {i} difiere tras round-trip de guardado.");
                }
            }
            finally
            {
                if (System.IO.File.Exists(policyPath))
                {
                    System.IO.File.Delete(policyPath);
                }
            }
        }
    }
}
