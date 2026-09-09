using System;
using System.Linq;
using Neuraval.Evolution.MarioBridge;
using Neuraval.Evolution.Neat;
using Xunit;

namespace Neuraval.Tests
{
    // Cubre la Fase 4 del roadmap: "Humano -> Imitation Learning -> politica
    // inicial -> NEAT (mutacion+crossover) -> Evaluacion -> nueva generacion".
    // El roadmap decia que este pipeline "ya esta soportado" via --learn y
    // --seed-policy, asi que lo que falta no es codigo nuevo sino verificar
    // que el enganche entre ambos mundos (red densa entrenada por
    // backprop <-> genoma NEAT evaluado por Genome.Evaluate) preserva el
    // comportamiento aprendido. Al escribir esos tests aparecio un bug real:
    // MarioPolicyNetwork.Forward usa Sigmoid en la capa de salida (asi la
    // entreno MarioImitationTrainer, con BCE, y asi lo espera
    // MarioAgentOutput.ButtonThreshold = 0.5f), pero NeatGenome.Evaluate
    // aplicaba Tanh a TODOS los nodos no-input, incluidos los de salida. Un
    // genoma sembrado desde una politica imitada perdia por completo el
    // comportamiento demostrado por el humano en cuanto NEAT lo evaluaba,
    // porque Tanh y Sigmoid tienen rangos y puntos medios distintos
    // ([-1,1]/0 vs [0,1]/0.5). Se corrigio NeatGenome.Evaluate para que los
    // nodos de salida usen Sigmoid (igual que MarioPolicyNetwork), dejando
    // Tanh solo para nodos ocultos. Estos tests fijan ese contrato.
    public class MarioSeedPolicyPipelineTests
    {
        private static float[] BuildSampleInput(int length, int seed)
        {
            var random = new Random(seed);
            var input = new float[length];
            for (var i = 0; i < length; i++)
            {
                // Rango similar al que produce MarioStateEncoder (mayormente
                // -1..1 con algunos 0/1), no hace falta reproducirlo exacto:
                // lo que se prueba es la fidelidad de la conversion, no el
                // encoder.
                input[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }

            return input;
        }

        [Fact]
        public void NeatGenome_OutputNodes_UseSigmoid_LikeMarioPolicyNetwork()
        {
            // Regresion directa del bug: con Tanh en la salida, Evaluate()
            // podia devolver valores negativos. MarioAgentOutput compara
            // contra ButtonThreshold=0.5f asumiendo semantica de Sigmoid
            // (rango 0..1). Si esto no se cumple, ningun boton salvo
            // Left/Right (que comparan entre si) podria presionarse de forma
            // confiable para pesos negativos o cercanos a cero, que es
            // exactamente el punto de partida tipico de un genoma random.
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
            // Este es el corazon de --seed-policy: TryLoadSeedGenome llama a
            // policy.AsGenome(tracker) y ese genoma pasa a ser el agente 0
            // de la poblacion NEAT. Si Evaluate() no reproduce Forward(), la
            // politica imitada "semilla" no aporta nada util: NEAT arranca
            // de un comportamiento distinto (y peor, no relacionado) al que
            // el humano demostro.
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

                // Ademas de los valores crudos, la decision de boton
                // (>= 0.5) tiene que coincidir: es lo que --play/--learn
                // realmente usa para actuar.
                Assert.Equal(MarioAgentOutput.ToAction(expected), MarioAgentOutput.ToAction(actual));
            }
        }

        [Fact]
        public void SeededGenome_SurvivesOneNeatGeneration_AndKeepsShapeAndSigmoidOutputs()
        {
            // Simula lo que hace RunTrainingModeCore cuando arranca sin
            // checkpoint pero con --seed-policy: agente 0 = genoma de la
            // politica imitada, resto = genomas random; despues corre NEAT
            // (mutacion/crossover) normalmente. Este test corre una
            // "generacion" completa contra NeatEvolutionStrategy con fitness
            // ficticio y verifica que el pipeline no se rompe y que los
            // genomas resultantes siguen produciendo salidas validas
            // (Sigmoid en 0..1) sin importar que mezcla de padres los haya
            // producido.
            const int populationSize = 12;
            const int hiddenSize = 6;

            var random = new Random(555);
            var tracker = new NeatInnovationTracker(MarioAgent.InputCount + MarioAgent.OutputCount + 1);

            var policy = MarioPolicyNetwork.Create(MarioAgent.InputCount, MarioAgent.OutputCount, random, hiddenSize);
            var seedGenome = policy.AsGenome(tracker);

            // Igual que TryLoadSeedGenome: adelantar el tracker para que las
            // mutaciones/crossovers posteriores no choquen con los
            // innovation numbers ya usados por el genoma semilla.
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

            // Fitness ficticio: el agente semilla (indice 0) es el mejor,
            // como se espera de una politica ya entrenada por imitacion
            // frente a genomas random recien creados.
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

                // No debe tirar: mismo contrato que usa MarioAgent.Decide.
                _ = MarioAgentOutput.ToAction(output);
            }
        }

        [Fact]
        public void SavedPolicy_RoundTrip_ThenSeeded_MatchesForwardBeforeSaving()
        {
            // Cierra el circuito completo de --learn -> --seed-policy: la
            // politica se guarda en disco (mario_policy.navm), se vuelve a
            // cargar (como hace TryLoadSeedGenome via MarioPolicyNetwork.Load)
            // y solo despues se convierte a genoma. Verifica que el
            // round-trip de guardado no introduce ninguna perdida que
            // AsGenome no cubra.
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
