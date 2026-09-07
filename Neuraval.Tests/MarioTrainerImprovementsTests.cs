using System.Collections.Generic;
using Neuraval.Evolution.MarioBridge;
using Xunit;

namespace Neuraval.Tests
{
    public class MarioTrainerImprovementsTests
    {
        [Fact]
        public void RewardWeighting_MapsRewardsToWeightsInOneToThreeRange()
        {
            Assert.Equal(1f, MarioRewardWeighting.Compute(0f, 0f, 10f, 2f));
            Assert.Equal(3f, MarioRewardWeighting.Compute(10f, 0f, 10f, 2f));
            Assert.Equal(2f, MarioRewardWeighting.Compute(5f, 0f, 10f, 2f));

            Assert.Equal(1f, MarioRewardWeighting.Compute(5f, 0f, 0f, 2f));
            Assert.Equal(1f, MarioRewardWeighting.Compute(5f, -3f, 7f, 0f));

            Assert.True(MarioRewardWeighting.Compute(8f, -2f, 10f, 2f) > MarioRewardWeighting.Compute(0f, -2f, 10f, 2f));
        }

        [Fact]
        public void ImitationTrainer_DrivesPolicyTowardTargetMask()
        {
            var seed = 1234;
            var random = new System.Random(seed);
            var inputCount = 8;
            var sampleCount = 120;

            var dataset = new MarioDataset(inputCount, MarioAgentOutput.Count);
            var inputs = new float[sampleCount][];

            for (var i = 0; i < sampleCount; i++)
            {
                var input = new float[inputCount];
                for (var j = 0; j < inputCount; j++)
                {
                    input[j] = random.NextSingle();
                }

                inputs[i] = input;
                var reward = i < sampleCount / 2 ? 1f : 0f;
                dataset.Samples.Add(new MarioDatasetSample(
                    i,
                    LevelIndex: 0,
                    input,
                    SnesButton.Right | SnesButton.A,
                    reward,
                    Done: false));
            }

            var trainer = new MarioImitationTrainer(dataset, epochs: 100);
            var policy = trainer.Train(new System.Random(seed));

            for (var i = 0; i < 5; i++)
            {
                var output = policy.Forward(inputs[(i * 17) % sampleCount]);
                Assert.True(output[MarioAgentOutput.RightIndex] > 0.5f, $"Right deberia activarse (got {output[MarioAgentOutput.RightIndex]:F4})");
                Assert.True(output[MarioAgentOutput.AIndex] > 0.5f, $"A deberia activarse (got {output[MarioAgentOutput.AIndex]:F4})");
                Assert.True(output[MarioAgentOutput.LeftIndex] < 0.5f, $"Left no deberia activarse (got {output[MarioAgentOutput.LeftIndex]:F4})");
                Assert.True(output[MarioAgentOutput.BIndex] < 0.5f, $"B no deberia activarse (got {output[MarioAgentOutput.BIndex]:F4})");
            }
        }
    [Fact]
        public void SpriteNames_KnownEnemyGetsHumanReadableName()
        {
            Assert.Equal("Goomba", MarioSpriteNames.Name(0x0F));
            Assert.Equal("Koopa verde", MarioSpriteNames.Name(0x04));
            Assert.Equal("Bala Bill", MarioSpriteNames.Name(0x1C));
            Assert.Equal("sprite $E7", MarioSpriteNames.Name(0xE7));
        }
    }
}