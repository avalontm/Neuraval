using System.IO;
using System.Text.Json;
using Neuraval.Evolution.Neat;
using Neuraval.Evolution.Serialization;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioNeatModelHeader
    {
        public const string KindValue = "neat-model";
        public string Kind { get; set; } = KindValue;
        public int FormatVersion { get; set; }
        public int InputCount { get; set; }
        public int OutputCount { get; set; }
        public DateTime SavedAtUtc { get; set; }
    }

    public static class MarioNeatModelStore
    {
        public const int CurrentFormatVersion = 1;

        private static readonly JsonSerializerOptions HeaderJsonOptions = new() { WriteIndented = false };

        public static void Save(string filePath, NeatGenome genome)
        {
            var header = new MarioNeatModelHeader
            {
                FormatVersion = CurrentFormatVersion,
                InputCount = genome.InputCount,
                OutputCount = genome.OutputCount,
                SavedAtUtc = DateTime.UtcNow
            };

            byte[] body;
            using (var bodyStream = new MemoryStream())
            {
                using (var writer = new BinaryWriter(bodyStream))
                {
                    MarioGenomeSerializer.Write(writer, genome);
                }

                body = bodyStream.ToArray();
            }

            NavmBinarySerializer.Save(filePath, JsonSerializer.Serialize(header, HeaderJsonOptions), body);
        }

        public static NeatGenome? Load(string filePath)
        {
            var content = NavmBinarySerializer.Load(filePath);
            if (content == null)
            {
                return null;
            }

            MarioNeatModelHeader? header;
            try
            {
                header = JsonSerializer.Deserialize<MarioNeatModelHeader>(content.HeaderJson);
            }
            catch (JsonException)
            {
                return null;
            }

            if (header == null || header.Kind != MarioNeatModelHeader.KindValue)
            {
                return null;
            }

            using var stream = new MemoryStream(content.Body);
            using var reader = new BinaryReader(stream);
            var genome = MarioGenomeSerializer.Read(reader);

            if (genome.InputCount != MarioAgent.InputCount || genome.OutputCount != MarioAgent.OutputCount)
            {
                return null;
            }

            return genome;
        }
    }
}