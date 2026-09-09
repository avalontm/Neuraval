using Neuraval.Evolution.Neat;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioAgent : IAgent<SnesState, SnesAction>, INeatAgent<MarioAgent>
    {
        public const int InputCount = MarioStateEncoder.StackedInputCount;
        public const int OutputCount = MarioAgentOutput.Count;

        private readonly MarioEncoderStack _history = new();

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

        public void ResetHistory()
        {
            _history.Reset();
        }

        public SnesAction Decide(SnesState state)
        {
            var input = _history.Encode(state);
            var output = Genome.Evaluate(input);
            return MarioAgentOutput.ToAction(output);
        }
    }
}