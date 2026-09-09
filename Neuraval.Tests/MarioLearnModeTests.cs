using System;
using System.IO;
using Neuraval.Evolution.MarioBridge;
using Xunit;

namespace Neuraval.Tests
{
    // Cubre el pipeline que arma "--learn": MarioDatasetRecorder (grabar al
    // humano) -> MarioDatasetLoader (releer sin reconectar BizHawk) ->
    // MarioImitationTrainer (entrenar) -> MarioPolicyNetwork (jugar), que es
    // exactamente la secuencia que RunLearnMode ejecuta dentro del mismo
    // proceso sin volver a invocar dotnet run entre pasos. No abre un socket
    // real hacia BizHawk (eso requeriria el emulador corriendo), pero valida
    // que grabar y despues entrenar en el mismo proceso produce una politica
    // cargable y jugable, igual que el flujo manual --capture + --imitate.
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
            // Simula la fase 1 de --learn: un humano "jugando" corriendo a la
            // derecha, grabado frame a frame con MarioDatasetRecorder (lo
            // mismo que hace RunLearnMode con cada estado que llega de
            // BizHawk).
            var datasetPath = Path.Combine(Path.GetTempPath(), $"mario_learn_dataset_{Guid.NewGuid():N}.navm");
            var policyPath = Path.Combine(Path.GetTempPath(), $"mario_learn_policy_{Guid.NewGuid():N}.navm");

            try
            {
                const int rightButton = 0x01; // bit "Right" en controller1; solo importa que sea consistente.
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

                // Fase 2 de --learn: releer el dataset recien grabado (sin
                // reconectar BizHawk) y entrenar la politica por imitacion,
                // exactamente como hace RunLearnMode apenas se corta la
                // grabacion con el primer Ctrl+C.
                var dataset = MarioDatasetLoader.Load(datasetPath);
                Assert.NotNull(dataset);
                Assert.Equal(200, dataset!.Samples.Count);
                Assert.Equal(MarioAgent.InputCount, dataset.InputCount);
                Assert.Equal(MarioAgent.OutputCount, dataset.OutputCount);

                var trainer = new MarioImitationTrainer(dataset, epochs: 2);
                var policy = trainer.Train(new Random(1234));
                policy.Save(policyPath);

                // Fase 3 de --learn: la politica recien entrenada tiene que
                // poder cargarse y usarse para jugar (--play), sin necesitar
                // recapturar ni reentrenar nada.
                var loaded = MarioPolicyNetwork.Load(policyPath);
                Assert.NotNull(loaded);

                var sampleInput = MarioStateEncoder.Encode(BuildState(0, 150, rightButton));
                var output = loaded!.Forward(sampleInput);
                Assert.Equal(MarioAgent.OutputCount, output.Length);
                foreach (var probability in output)
                {
                    Assert.InRange(probability, 0f, 1f);
                }

                // No debe tirar: convierte la salida de la red en una accion
                // valida, igual que hace el loop de --play/--learn en cada
                // frame.
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
            // RunLearnMode reutiliza la MISMA conexion TCP para la fase de
            // grabacion y la fase de juego automatico: cuando termina de
            // grabar, BizHawk ya esta esperando NUESTRA respuesta al ultimo
            // estado que mandamos leer (no va a mandar un estado "de
            // arranque" nuevo por su cuenta). Sin MarkConnected(), el primer
            // Reset() intentaria leer un estado extra antes de mandar RESET
            // y se quedaria esperando datos que nunca llegan (deadlock). Este
            // test no abre un socket real, pero fija el contrato de
            // MarkConnected() via reflection sobre el campo privado
            // "_connected", que es exactamente lo que Reset() consulta antes
            // de decidir si hace una lectura extra.
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
