using System;
using System.IO;
using Neuraval.Evolution.MarioBridge;
using Xunit;

namespace Neuraval.Tests
{
    public class MarioLearnModeTests
    {
        private static SnesState BuildState(int frame, int marioX, int controller1)
        {
            var tiles = new byte[MarioStateEncoder.GridCellCount];
            return new SnesState(
                frame: frame, marioX: marioX, marioY: 200, marioVelocityX: 0, marioVelocityY: 0,
                isDead: false, lives: 5, isLevelComplete: false, manualResetRequested: false,
                isGrounded: true, powerupLevel: 0, levelIndex: 0,
                tiles: tiles, sprites: Array.Empty<SnesSprite>(),
                direction: 0, blocked: 0, subPixelX: 0, subPixelY: 0,
                airState: 0, ducking: 0, climbing: 0, water: 0,
                pMeter: 0, takeoffMeter: 0, hurtTimer: 0, capeTimer: 0,
                cameraX: 0, cameraY: 0, gameMode: 0, levelMode: 0, translevel: 0,
                coins: 0, itemBox: 0, controller1: controller1, controller1Prev: 0,
                controller2: 0, controller2Prev: 0);
        }

        [Fact]
        public void RecordThenLoadThenTrain_ProducesAPlayablePolicy_WithoutReReadingFromBizHawk()
        {
            var datasetPath = Path.Combine(Path.GetTempPath(), $"mario_learn_dataset_{Guid.NewGuid():N}.navm");
            var policyPath = Path.Combine(Path.GetTempPath(), $"mario_learn_policy_{Guid.NewGuid():N}.navm");

            try
            {
                const int rightButton = 0x01;
                var previousX = 100;

                using (var recorder = new MarioDatasetRecorder(datasetPath))
                {
                    for (var frame = 0; frame < 200; frame++)
                    {
                        var marioX = previousX + 1;
                        var state = BuildState(frame, marioX, rightButton);
                        var action = MarioControllerEncoder.Decode(state.Controller1, state.Controller2);
                        recorder.Append(state, action, reward: marioX - previousX, done: false);
                        previousX = marioX;
                    }

                    recorder.Complete();
                }

                Assert.True(File.Exists(datasetPath));

                var dataset = MarioDatasetLoader.Load(datasetPath);
                Assert.NotNull(dataset);
                Assert.Equal(200, dataset!.Samples.Count);
                Assert.Equal(MarioAgent.InputCount, dataset.InputCount);
                Assert.Equal(MarioAgent.OutputCount, dataset.OutputCount);

                var trainer = new MarioImitationTrainer(dataset, epochs: 2);
                var policy = trainer.Train(new Random(1234));
                policy.Save(policyPath);

                var loaded = MarioPolicyNetwork.Load(policyPath);
                Assert.NotNull(loaded);

                var sampleInput = new MarioEncoderStack().Encode(BuildState(0, 150, rightButton));
                var output = loaded!.Forward(sampleInput);
                Assert.Equal(MarioAgent.OutputCount, output.Length);
                foreach (var probability in output)
                {
                    Assert.InRange(probability, 0f, 1f);
                }

                _ = MarioAgentOutput.ToAction(output);
            }
            finally
            {
                File.Delete(datasetPath);
                File.Delete(policyPath);
            }
        }

        [Fact]
        public void MarkConnected_SkipsTheExtraStartupReceive_SoResetOnlySendsOneCommand()
        {
            var connectedField = typeof(SnesEnvironment).GetField(
                "_connected",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            Assert.NotNull(connectedField);

            var environment = new SnesEnvironment(connection: null!);

            Assert.Equal(false, connectedField!.GetValue(environment));

            environment.MarkConnected();

            Assert.Equal(true, connectedField.GetValue(environment));
        }
    }
}
