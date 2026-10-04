using System;
using System.IO;
using Neuraval.Evolution.MarioBridge;
using Xunit;

namespace Neuraval.Tests
{
    public class MarioPlayModeTests
    {
        [Fact]
        public void Load_ReturnsNull_WhenInputOrOutputCountDoesNotMatchCurrentEncoder()
        {
            var path = Path.Combine(Path.GetTempPath(), $"mario_policy_mismatch_{Guid.NewGuid():N}.navm");
            try
            {
                var staleInputCount = Math.Max(1, MarioAgent.InputCount - 1);
                var stalePolicy = MarioPolicyNetwork.Create(staleInputCount, MarioAgent.OutputCount, new Random(1));
                stalePolicy.Save(path);

                var loaded = MarioPolicyNetwork.Load(path);

                Assert.Null(loaded);
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void Load_ReturnsNull_WhenFileDoesNotExist()
        {
            var missingPath = Path.Combine(Path.GetTempPath(), $"mario_policy_missing_{Guid.NewGuid():N}.navm");

            var loaded = MarioPolicyNetwork.Load(missingPath);

            Assert.Null(loaded);
        }

        [Fact]
        public void PlayPipeline_LoadsSavedPolicyAndProducesAValidAction()
        {
            var path = Path.Combine(Path.GetTempPath(), $"mario_policy_play_{Guid.NewGuid():N}.navm");
            try
            {
                var trained = MarioPolicyNetwork.Create(MarioAgent.InputCount, MarioAgent.OutputCount, new Random(42));
                trained.Save(path);

                var loaded = MarioPolicyNetwork.Load(path);
                Assert.NotNull(loaded);
                Assert.Equal(MarioAgent.InputCount, loaded!.InputCount);
                Assert.Equal(MarioAgent.OutputCount, loaded.OutputCount);

                var input = new float[MarioAgent.InputCount];
                for (var i = 0; i < input.Length; i++)
                {
                    input[i] = (i % 7) / 7f;
                }

                var output = loaded.Forward(input);
                Assert.Equal(MarioAgent.OutputCount, output.Length);

                var action = MarioAgentOutput.ToAction(output);
                var pressesLeftAndRight = action.IsPressed(SnesButton.Left) && action.IsPressed(SnesButton.Right);
                Assert.False(pressesLeftAndRight);
            }
            finally
            {
                File.Delete(path);
            }
        }
    }
}
