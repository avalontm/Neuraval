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
            try
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

                var headerJson = JsonSerializer.Serialize(header, HeaderJsonOptions);

                TransientFileIoRetry.Run(
                    () => NavmBinarySerializer.Save(filePath, headerJson, body),
                    onRetry: (attempt, maxRetries, ex) => Console.WriteLine(
                        $"Modelo ({filePath}): el archivo esta en uso por otro proceso ({ex.Message}); reintentando ({attempt}/{maxRetries})..."));
            }
            catch (Exception ex)
            {
                // No dejamos que un fallo al guardar el "mejor modelo" tire
                // abajo el entrenamiento: el checkpoint principal (que SI
                // tiene reintentos y ya guardo bien) sigue teniendo el mismo
                // genoma, y en el proximo generation se vuelve a intentar
                // guardar este archivo.
                Console.WriteLine($"No se pudo guardar el modelo ({filePath}): {ex.GetType().Name}: {ex.Message}");
            }
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