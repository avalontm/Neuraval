using System;
using System.IO;
using Neuraval.Evolution.MarioBridge;
using Xunit;

namespace Neuraval.Tests
{
    // Cubre el pipeline que usa "--play": Load(policy) -> Forward(input) ->
    // ToAction(output). No abre un socket real hacia BizHawk (eso requeriria
    // el emulador corriendo), pero valida exactamente lo mismo que
    // RunPlayMode hace con cada frame, y el guard rail que evita que arranque
    // con una politica que no coincide con el encoder/salidas actuales.
    public class MarioPlayModeTests
    {
        [Fact]
        public void Load_ReturnsNull_WhenInputOrOutputCountDoesNotMatchCurrentEncoder()
        {
            var path = Path.Combine(Path.GetTempPath(), $"mario_policy_mismatch_{Guid.NewGuid():N}.navm");
            try
            {
                // Politica valida para un encoder viejo, con menos entradas que
                // el actual MarioAgent.InputCount: simula recapturar/reentrenar
                // despues de un cambio de protocolo (como los que ya documenta
                // el README, ej. GridRadius 6->8).
                var staleInputCount = Math.Max(1, MarioAgent.InputCount - 1);
                var stalePolicy = MarioPolicyNetwork.Create(staleInputCount, MarioAgent.OutputCount, new Random(1));
                stalePolicy.Save(path);

                var loaded = MarioPolicyNetwork.Load(path);

                // Este es exactamente el caso que RunPlayMode detecta antes de
                // conectarse a BizHawk y por el que se niega a arrancar con un
                // mensaje explicando que hay que recapturar/reentrenar.
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

                // Mismos pasos que RunPlayMode ejecuta en cada frame: cargar la
                // politica una vez, y despues por cada estado recibido de
                // BizHawk, Forward(input) -> ToAction(output).
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

                // ToAction no debe tirar ni devolver un mix invalido de
                // izquierda+derecha simultaneo (MarioAgentOutput ya resuelve
                // ese empate quedandose con uno solo de los dos).
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
