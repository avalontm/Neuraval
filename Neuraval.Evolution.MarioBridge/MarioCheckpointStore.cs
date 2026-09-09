using System.IO;
using System.Text.Json;
using Neuraval.Evolution.Neat;
using Neuraval.Evolution.Serialization;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioCheckpoint
    {
        public int Generation { get; set; }
        public float BestFitnessEver { get; set; }
        public NeatGenome? BestGenomeEver { get; set; }
        public List<NeatGenome> Genomes { get; set; } = new List<NeatGenome>();

        // Nullable a proposito: checkpoints guardados antes de Fase 3 no
        // tienen estos campos en su header JSON, y System.Text.Json los
        // deserializa como null sin tirar excepcion. Un null significa
        // "sin progreso de curriculum guardado todavia" y el llamador
        // (Program.cs) lo interpreta como arrancar en el tramo 0.
        public int? CurriculumStageIndex { get; set; }
        public int? CurriculumGenerationsAtStage { get; set; }
    }

    public sealed class MarioCheckpointHeader
    {
        public const string KindValue = "checkpoint";
        public string Kind { get; set; } = KindValue;
        public int FormatVersion { get; set; }
        public int InputCount { get; set; }
        public int OutputCount { get; set; }
        public int Generation { get; set; }
        public float BestFitnessEver { get; set; }
        public DateTime SavedAtUtc { get; set; }
        public int? CurriculumStageIndex { get; set; }
        public int? CurriculumGenerationsAtStage { get; set; }
    }

    public static class MarioCheckpointStore
    {
        public const int CurrentFormatVersion = 1;

        private static readonly string DefaultSaveFilePath = Path.Combine("checkpoints", "mario_checkpoint.navm");
        private static readonly JsonSerializerOptions HeaderJsonOptions = new() { WriteIndented = false };

        public static string SaveFilePath { get; set; } = DefaultSaveFilePath;

        private static string BackupFilePath => SaveFilePath + ".bak";

        public static MarioCheckpoint? Load()
        {
            var checkpoint = TryLoadFrom(SaveFilePath, "principal");
            if (checkpoint != null)
            {
                return checkpoint;
            }

            if (File.Exists(BackupFilePath))
            {
                Console.WriteLine("Probando con el backup automatico (.bak)...");
                checkpoint = TryLoadFrom(BackupFilePath, "backup");
                if (checkpoint != null)
                {
                    Console.WriteLine("Se recupero el progreso desde el backup.");
                }
            }

            return checkpoint;
        }

        private static MarioCheckpoint? TryLoadFrom(string path, string label)
        {
            try
            {
                if (!File.Exists(path))
                {
                    return null;
                }

                var content = NavmBinarySerializer.Load(path);
                if (content == null)
                {
                    Console.WriteLine($"El checkpoint {label} ({path}) no es un archivo NAVM valido o esta corrupto; se ignora.");
                    ArchiveUnusable(path, "no-valid");
                    return null;
                }

                var header = DeserializeHeader(content.HeaderJson);
                if (header == null || header.Kind != MarioCheckpointHeader.KindValue)
                {
                    Console.WriteLine($"El checkpoint {label} ({path}) no tiene un encabezado de checkpoint valido; se ignora.");
                    ArchiveUnusable(path, "header-invalido");
                    return null;
                }

                var checkpoint = new MarioCheckpoint
                {
                    Generation = header.Generation,
                    BestFitnessEver = header.BestFitnessEver,
                    CurriculumStageIndex = header.CurriculumStageIndex,
                    CurriculumGenerationsAtStage = header.CurriculumGenerationsAtStage
                };

                using (var stream = new MemoryStream(content.Body))
                using (var reader = new BinaryReader(stream))
                {
                    var hasBestGenome = reader.ReadBoolean();
                    checkpoint.BestGenomeEver = hasBestGenome ? MarioGenomeSerializer.Read(reader) : null;

                    var genomeCount = reader.ReadInt32();
                    for (var i = 0; i < genomeCount; i++)
                    {
                        checkpoint.Genomes.Add(MarioGenomeSerializer.Read(reader));
                    }
                }

                if (header.InputCount != MarioAgent.InputCount || header.OutputCount != MarioAgent.OutputCount)
                {
                    Console.WriteLine(
                        $"El checkpoint {label} ({path}) es incompatible con la red actual: " +
                        $"fue guardado con {header.InputCount} entradas / {header.OutputCount} salidas, " +
                        $"pero el codigo actual usa {MarioAgent.InputCount} entradas / {MarioAgent.OutputCount} salidas. " +
                        "Esto pasa cuando se cambia la topologia de la red (por ejemplo, se agrega un boton nuevo o " +
                        "una senal de entrada nueva) sin correr con --reset. Se archiva el checkpoint viejo (no se " +
                        "pierde) y se arranca una poblacion nueva compatible.");
                    ArchiveUnusable(path, $"incompatible-{header.InputCount}in-{header.OutputCount}out");
                    return null;
                }

                Console.WriteLine($"Checkpoint {label} valido: generacion {checkpoint.Generation}, guardado {header.SavedAtUtc:u} UTC.");
                return checkpoint;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"El checkpoint {label} ({path}) esta corrupto o incompleto ({ex.GetType().Name}: {ex.Message}); se ignora.");
                ArchiveUnusable(path, "corrupto");
                return null;
            }
        }

        public static void Save(MarioCheckpoint checkpoint)
        {
            try
            {
                var fullPath = Path.GetFullPath(SaveFilePath);
                var directory = Path.GetDirectoryName(fullPath);
                if (!string.IsNullOrEmpty(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                var header = new MarioCheckpointHeader
                {
                    FormatVersion = CurrentFormatVersion,
                    InputCount = MarioAgent.InputCount,
                    OutputCount = MarioAgent.OutputCount,
                    Generation = checkpoint.Generation,
                    BestFitnessEver = checkpoint.BestFitnessEver,
                    SavedAtUtc = DateTime.UtcNow,
                    CurriculumStageIndex = checkpoint.CurriculumStageIndex,
                    CurriculumGenerationsAtStage = checkpoint.CurriculumGenerationsAtStage
                };

                byte[] body;
                using (var bodyStream = new MemoryStream())
                {
                    using (var writer = new BinaryWriter(bodyStream))
                    {
                        writer.Write(checkpoint.BestGenomeEver != null);
                        if (checkpoint.BestGenomeEver != null)
                        {
                            MarioGenomeSerializer.Write(writer, checkpoint.BestGenomeEver);
                        }

                        writer.Write(checkpoint.Genomes.Count);
                        foreach (var genome in checkpoint.Genomes)
                        {
                            MarioGenomeSerializer.Write(writer, genome);
                        }
                    }

                    body = bodyStream.ToArray();
                }

                var tempPath = SaveFilePath + ".tmp";
                var headerJson = JsonSerializer.Serialize(header, HeaderJsonOptions);

                TransientFileIoRetry.Run(
                    () =>
                    {
                        NavmBinarySerializer.Save(tempPath, headerJson, body);

                        if (File.Exists(SaveFilePath))
                        {
                            File.Replace(tempPath, SaveFilePath, BackupFilePath, ignoreMetadataErrors: true);
                        }
                        else
                        {
                            File.Move(tempPath, SaveFilePath, overwrite: true);
                        }
                    },
                    onRetry: (attempt, maxRetries, ex) => Console.WriteLine(
                        $"Checkpoint: el archivo esta en uso por otro proceso ({ex.Message}); reintentando ({attempt}/{maxRetries})..."));

                var savedSizeKb = new FileInfo(SaveFilePath).Length / 1024.0;
                Console.WriteLine($"Checkpoint guardado: generacion {checkpoint.Generation}, {checkpoint.Genomes.Count} genomas, {savedSizeKb:F1} KB.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"No se pudo guardar el checkpoint: {ex.GetType().Name}: {ex.Message}");
            }
        }

        public static void Delete()
        {
            try
            {
                if (File.Exists(SaveFilePath))
                {
                    File.Delete(SaveFilePath);
                }

                if (File.Exists(BackupFilePath))
                {
                    File.Delete(BackupFilePath);
                }
            }
            catch
            {
            }
        }

        private static MarioCheckpointHeader? DeserializeHeader(string json)
        {
            try
            {
                return JsonSerializer.Deserialize<MarioCheckpointHeader>(json);
            }
            catch (JsonException)
            {
                return null;
            }
        }

        private static void ArchiveUnusable(string path, string reason)
        {
            try
            {
                if (!File.Exists(path))
                {
                    return;
                }

                var timestamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss");
                var archivedPath = $"{path}.{reason}.{timestamp}";
                File.Move(path, archivedPath, overwrite: true);
                Console.WriteLine($"Archivo movido a: {archivedPath}");
            }
            catch
            {
            }
        }
    }
}