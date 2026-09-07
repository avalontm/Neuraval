using Neuraval.Evolution.Neat;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioAgent : IAgent<SnesState, SnesAction>, INeatAgent<MarioAgent>
    {
        public const int InputCount = MarioStateEncoder.InputCount;
        public const int OutputCount = MarioAgentOutput.Count;

        public NeatGenome Genome { get; }

        public MarioAgent(NeatGenome genome)
        {
            Genome = genome;
        }

        public static MarioAgent CreateRandom(Random random, NeatInnovationTracker tracker)
        {
            return new MarioAgent(NeatGenome.CreateInitial(InputCount, OutputCount, random, tracker));
        }

        public MarioAgent WithGenome(NeatGenome genome)
        {
            return new MarioAgent(genome);
        }

        public SnesAction Decide(SnesState state)
        {
            var input = MarioStateEncoder.Encode(state);
            var output = Genome.Evaluate(input);
            return MarioAgentOutput.ToAction(output);
        }
    }
}