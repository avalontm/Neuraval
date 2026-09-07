using System.IO;
using System.Text.Json;
using Neuraval.Evolution.Neat;
using Neuraval.Evolution.Serialization;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioPolicyHeader
    {
        public const string KindValue = "policy";
        public string Kind { get; set; } = KindValue;
        public int FormatVersion { get; set; }
        public int InputCount { get; set; }
        public int OutputCount { get; set; }
        public int HiddenSize { get; set; }
        public DateTime SavedAtUtc { get; set; }
    }

    public sealed class MarioPolicyNetwork
    {
        public const int DefaultHiddenSize = 128;
        public const int CurrentFormatVersion = 1;

        private static readonly JsonSerializerOptions HeaderJsonOptions = new() { WriteIndented = false };

        public int InputCount { get; }
        public int OutputCount { get; }
        public int HiddenSize { get; }

        internal float[] WeightsIn { get; }
        internal float[] BiasHidden { get; }
        internal float[] WeightsOut { get; }
        internal float[] BiasOut { get; }

        private MarioPolicyNetwork(int inputCount, int outputCount, int hiddenSize, float[] weightsIn, float[] biasHidden, float[] weightsOut, float[] biasOut)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            HiddenSize = hiddenSize;
            WeightsIn = weightsIn;
            BiasHidden = biasHidden;
            WeightsOut = weightsOut;
            BiasOut = biasOut;
        }

        public static MarioPolicyNetwork Create(int inputCount, int outputCount, Random random, int hiddenSize = DefaultHiddenSize)
        {
            var scaleIn = 1f / MathF.Sqrt(inputCount + hiddenSize);
            var scaleOut = 1f / MathF.Sqrt(hiddenSize + outputCount);

            var weightsIn = new float[hiddenSize * inputCount];
            var biasHidden = new float[hiddenSize];
            var weightsOut = new float[outputCount * hiddenSize];
            var biasOut = new float[outputCount];

            for (var i = 0; i < weightsIn.Length; i++)
            {
                weightsIn[i] = NextUniform(random, scaleIn);
            }

            for (var i = 0; i < biasHidden.Length; i++)
            {
                biasHidden[i] = NextUniform(random, scaleIn);
            }

            for (var i = 0; i < weightsOut.Length; i++)
            {
                weightsOut[i] = NextUniform(random, scaleOut);
            }

            for (var i = 0; i < biasOut.Length; i++)
            {
                biasOut[i] = NextUniform(random, scaleOut);
            }

            return new MarioPolicyNetwork(inputCount, outputCount, hiddenSize, weightsIn, biasHidden, weightsOut, biasOut);
        }

        public float[] Forward(float[] input)
        {
            var hidden = new float[HiddenSize];

            for (var h = 0; h < HiddenSize; h++)
            {
                var sum = BiasHidden[h];
                for (var i = 0; i < InputCount; i++)
                {
                    sum += WeightsIn[h * InputCount + i] * input[i];
                }

                hidden[h] = MathF.Tanh(sum);
            }

            var output = new float[OutputCount];

            for (var o = 0; o < OutputCount; o++)
            {
                var sum = BiasOut[o];
                for (var h = 0; h < HiddenSize; h++)
                {
                    sum += WeightsOut[o * HiddenSize + h] * hidden[h];
                }

                output[o] = Sigmoid(sum);
            }

            return output;
        }

        public NeatGenome AsGenome(NeatInnovationTracker tracker)
        {
            var nodes = new List<NeatNodeGene>();
            for (var i = 0; i < InputCount; i++)
            {
                nodes.Add(new NeatNodeGene(i, NeatNodeType.Input));
            }

            var biasId = InputCount;
            nodes.Add(new NeatNodeGene(biasId, NeatNodeType.Bias));

            var hiddenStart = biasId + 1;
            for (var h = 0; h < HiddenSize; h++)
            {
                nodes.Add(new NeatNodeGene(hiddenStart + h, NeatNodeType.Hidden));
            }

            var outputStart = hiddenStart + HiddenSize;
            for (var o = 0; o < OutputCount; o++)
            {
                nodes.Add(new NeatNodeGene(outputStart + o, NeatNodeType.Output));
            }

            var connections = new List<NeatConnectionGene>();

            for (var h = 0; h < HiddenSize; h++)
            {
                for (var i = 0; i < InputCount; i++)
                {
                    var innovation = tracker.GetOrCreateConnectionInnovation(i, hiddenStart + h);
                    connections.Add(new NeatConnectionGene(i, hiddenStart + h, WeightsIn[h * InputCount + i], true, innovation));
                }

                var biasToHiddenInnovation = tracker.GetOrCreateConnectionInnovation(biasId, hiddenStart + h);
                connections.Add(new NeatConnectionGene(biasId, hiddenStart + h, BiasHidden[h], true, biasToHiddenInnovation));
            }

            for (var o = 0; o < OutputCount; o++)
            {
                for (var h = 0; h < HiddenSize; h++)
                {
                    var innovation = tracker.GetOrCreateConnectionInnovation(hiddenStart + h, outputStart + o);
                    connections.Add(new NeatConnectionGene(hiddenStart + h, outputStart + o, WeightsOut[o * HiddenSize + h], true, innovation));
                }

                var biasToOutputInnovation = tracker.GetOrCreateConnectionInnovation(biasId, outputStart + o);
                connections.Add(new NeatConnectionGene(biasId, outputStart + o, BiasOut[o], true, biasToOutputInnovation));
            }

            return new NeatGenome(InputCount, OutputCount, nodes, connections);
        }

        public void Save(string filePath)
        {
            var header = new MarioPolicyHeader
            {
                FormatVersion = CurrentFormatVersion,
                InputCount = InputCount,
                OutputCount = OutputCount,
                HiddenSize = HiddenSize,
                SavedAtUtc = DateTime.UtcNow
            };

            byte[] body;
            using (var bodyStream = new MemoryStream())
            {
                using (var writer = new BinaryWriter(bodyStream))
                {
                    WriteArray(writer, WeightsIn);
                    WriteArray(writer, BiasHidden);
                    WriteArray(writer, WeightsOut);
                    WriteArray(writer, BiasOut);
                }

                body = bodyStream.ToArray();
            }

            NavmBinarySerializer.Save(filePath, JsonSerializer.Serialize(header, HeaderJsonOptions), body);
        }

        public static MarioPolicyNetwork? Load(string filePath)
        {
            var content = NavmBinarySerializer.Load(filePath);
            if (content == null)
            {
                return null;
            }

            MarioPolicyHeader? header;
            try
            {
                header = JsonSerializer.Deserialize<MarioPolicyHeader>(content.HeaderJson);
            }
            catch (JsonException)
            {
                return null;
            }

            if (header == null || header.Kind != MarioPolicyHeader.KindValue)
            {
                return null;
            }

            if (header.InputCount != MarioAgent.InputCount || header.OutputCount != MarioAgent.OutputCount)
            {
                return null;
            }

            using var stream = new MemoryStream(content.Body);
            using var reader = new BinaryReader(stream);
            var weightsIn = ReadArray(reader, header.HiddenSize * header.InputCount);
            var biasHidden = ReadArray(reader, header.HiddenSize);
            var weightsOut = ReadArray(reader, header.OutputCount * header.HiddenSize);
            var biasOut = ReadArray(reader, header.OutputCount);

            return new MarioPolicyNetwork(header.InputCount, header.OutputCount, header.HiddenSize, weightsIn, biasHidden, weightsOut, biasOut);
        }

        private static float NextUniform(Random random, float scale)
        {
            return (random.NextSingle() * 2f - 1f) * scale;
        }

        private static float Sigmoid(float value)
        {
            return 1f / (1f + MathF.Exp(-value));
        }

        private static void WriteArray(BinaryWriter writer, float[] values)
        {
            foreach (var value in values)
            {
                writer.Write(value);
            }
        }

        private static float[] ReadArray(BinaryReader reader, int count)
        {
            var values = new float[count];
            for (var i = 0; i < count; i++)
            {
                values[i] = reader.ReadSingle();
            }

            return values;
        }
    }
}