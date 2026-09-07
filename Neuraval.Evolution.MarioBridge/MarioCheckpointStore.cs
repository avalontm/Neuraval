using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Neuraval.Evolution.Neat;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioCheckpoint
    {
        public int Generation { get; set; }
        public float BestFitnessEver { get; set; }
        public NeatGenome BestGenomeEver { get; set; }
        public List<NeatGenome> Genomes { get; set; } = new List<NeatGenome>();
    }

    public static class MarioCheckpointStore
    {
        // "NAVM" = Neural Avalon Model.
        //
        // v1: magic + version + generation + fitness + genomes. No metadata
        //     sobre la forma de la red (InputCount/OutputCount) a nivel de
        //     archivo -- solo dentro de cada genoma individual.
        // v2: agrega un bloque de metadata justo despues de la version, con
        //     InputCount/OutputCount/PopulationSize/fecha de guardado. Esto
        //     permite detectar ANTES de tocar un solo genoma si el checkpoint
        //     es compatible con la red que corre el codigo actual, en vez de
        //     descubrirlo a los golpes con un IndexOutOfRangeException a
        //     mitad de entrenamiento (que es lo que pasaba antes: un
        //     checkpoint con OutputCount viejo se cargaba "bien", pero
        //     Decide() reventaba al leer un boton que esos genomas nunca
        //     tuvieron, y el proceso moria en silencio con el archivo
        //     congelado en su ultimo tamano bueno).
        private const string MagicHeader = "NAVM";
        private const int FormatVersionLegacyV1 = 1;
        private const int FormatVersionCurrent = 2;

        private static readonly string DefaultSaveFilePath = Path.Combine("checkpoints", "mario_checkpoint.navm");

        // Relative by default, so "the checkpoint" is just a file you can zip up with the
        // project folder and drop on another machine. Program.cs can override this from
        // --checkpoint <path> if you want it on a pendrive, Dropbox, etc.
        public static string SaveFilePath { get; set; } = DefaultSaveFilePath;

        private static string BackupFilePath => SaveFilePath + ".bak";

        /// <summary>
        /// Carga el checkpoint y valida que sea compatible con la forma de red
        /// actual (MarioAgent.InputCount / MarioAgent.OutputCount). Si el
        /// archivo principal esta corrupto o es incompatible, intenta el
        /// backup automaticamente antes de rendirse. Nunca tira excepciones:
        /// en el peor caso devuelve null y el llamador arranca de cero, pero
        /// siempre deja un mensaje explicando por que.
        /// </summary>
        public static MarioCheckpoint Load()
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

        private static MarioCheckpoint TryLoadFrom(string path, string label)
        {
            try
            {
                if (!File.Exists(path))
                {
                    return null;
                }

                using var stream = File.OpenRead(path);
                using var reader = new BinaryReader(stream);

                var magic = new string(reader.ReadChars(MagicHeader.Length));
                if (magic != MagicHeader)
                {
                    Console.WriteLine($"El checkpoint {label} ({path}) no tiene el header NAVM esperado; se ignora.");
                    ArchiveUnusable(path, "header-invalido");
                    return null;
                }

                var version = reader.ReadInt32();

                int recordedInputCount;
                int recordedOutputCount;
                int recordedPopulationSize;
                DateTime? savedAtUtc = null;

                if (version == FormatVersionCurrent)
                {
                    recordedInputCount = reader.ReadInt32();
                    recordedOutputCount = reader.ReadInt32();
                    recordedPopulationSize = reader.ReadInt32();
                    savedAtUtc = new DateTime(reader.ReadInt64(), DateTimeKind.Utc);
                }
                else if (version == FormatVersionLegacyV1)
                {
                    // v1 no tiene metadata a nivel de archivo. La forma de la
                    // red se infiere del primer genoma una vez leido (mas
                    // abajo), asumiendo compatible hasta entonces.
                    recordedInputCount = -1;
                    recordedOutputCount = -1;
                    recordedPopulationSize = -1;
                }
                else
                {
                    Console.WriteLine($"El checkpoint {label} ({path}) tiene version de formato {version}, no soportada por este build; se ignora.");
                    ArchiveUnusable(path, $"version-{version}-no-soportada");
                    return null;
                }

                var checkpoint = new MarioCheckpoint
                {
                    Generation = reader.ReadInt32(),
                    BestFitnessEver = reader.ReadSingle()
                };

                var hasBestGenome = reader.ReadBoolean();
                checkpoint.BestGenomeEver = hasBestGenome ? ReadGenome(reader) : null;

                var genomeCount = reader.ReadInt32();
                for (var i = 0; i < genomeCount; i++)
                {
                    checkpoint.Genomes.Add(ReadGenome(reader));
                }

                // Para v1, no teniamos metadata de archivo: la sacamos del
                // primer genoma disponible (bestGenome o el primero de la
                // poblacion), que es lo mas parecido a "la forma con la que
                // se guardo esto".
                if (recordedInputCount < 0)
                {
                    var referenceGenome = checkpoint.BestGenomeEver ?? checkpoint.Genomes.FirstOrDefault();
                    recordedInputCount = referenceGenome?.InputCount ?? MarioAgent.InputCount;
                    recordedOutputCount = referenceGenome?.OutputCount ?? MarioAgent.OutputCount;
                }

                if (recordedInputCount != MarioAgent.InputCount || recordedOutputCount != MarioAgent.OutputCount)
                {
                    Console.WriteLine(
                        $"El checkpoint {label} ({path}) es incompatible con la red actual: " +
                        $"fue guardado con {recordedInputCount} entradas / {recordedOutputCount} salidas, " +
                        $"pero el codigo actual usa {MarioAgent.InputCount} entradas / {MarioAgent.OutputCount} salidas. " +
                        "Esto pasa cuando se cambia la topologia de la red (por ejemplo, se agrega un boton nuevo o " +
                        "una senal de entrada nueva) sin correr con --reset. Se archiva el checkpoint viejo (no se " +
                        "pierde) y se arranca una poblacion nueva compatible.");
                    ArchiveUnusable(path, $"incompatible-{recordedInputCount}in-{recordedOutputCount}out");
                    return null;
                }

                if (savedAtUtc.HasValue)
                {
                    Console.WriteLine($"Checkpoint {label} valido: generacion {checkpoint.Generation}, guardado {savedAtUtc.Value:u} UTC.");
                }

                return checkpoint;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"El checkpoint {label} ({path}) esta corrupto o incompleto ({ex.GetType().Name}: {ex.Message}); se ignora.");
                ArchiveUnusable(path, "corrupto");
                return null;
            }
        }

        /// <summary>
        /// Renombra (no borra) un checkpoint que no se puede usar, para que
        /// quede disponible por si alguien lo quiere inspeccionar despues,
        /// pero deje de interferir con la carga normal.
        /// </summary>
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
                // Si ni siquiera se puede archivar, seguimos: preferimos
                // arrancar de cero antes que colgar el entrenamiento.
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

                var tempPath = SaveFilePath + ".tmp";

                using (var stream = File.Create(tempPath))
                using (var writer = new BinaryWriter(stream))
                {
                    writer.Write(MagicHeader.ToCharArray());
                    writer.Write(FormatVersionCurrent);

                    // Metadata de v2: guardamos la forma exacta de red con la
                    // que se genero este archivo, para poder validar antes de
                    // leer un solo genoma la proxima vez que se cargue.
                    writer.Write(MarioAgent.InputCount);
                    writer.Write(MarioAgent.OutputCount);
                    writer.Write(checkpoint.Genomes.Count);
                    writer.Write(DateTime.UtcNow.Ticks);

                    writer.Write(checkpoint.Generation);
                    writer.Write(checkpoint.BestFitnessEver);

                    writer.Write(checkpoint.BestGenomeEver != null);
                    if (checkpoint.BestGenomeEver != null)
                    {
                        WriteGenome(writer, checkpoint.BestGenomeEver);
                    }

                    writer.Write(checkpoint.Genomes.Count);
                    foreach (var genome in checkpoint.Genomes)
                    {
                        WriteGenome(writer, genome);
                    }

                    writer.Flush();
                    stream.Flush(flushToDisk: true);
                }

                // Guardado atomico con backup automatico: File.Replace hace
                // "escribir el reemplazo, y solo si eso funciona, mover el
                // archivo viejo a BackupFilePath y poner el nuevo en su
                // lugar" como una sola operacion. Si el proceso muere a
                // mitad de un guardado (por ejemplo, se corta la luz o se
                // cierra BizHawk de golpe), el peor caso es que el .tmp
                // quede a medio escribir y se pise en el proximo intento --
                // el archivo principal y el .bak nunca quedan en un estado
                // a medio escribir.
                if (File.Exists(SaveFilePath))
                {
                    File.Replace(tempPath, SaveFilePath, BackupFilePath, ignoreMetadataErrors: true);
                }
                else
                {
                    File.Move(tempPath, SaveFilePath, overwrite: true);
                }

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

        private static void WriteGenome(BinaryWriter writer, NeatGenome genome)
        {
            writer.Write(genome.InputCount);
            writer.Write(genome.OutputCount);

            writer.Write(genome.Nodes.Count);
            foreach (var node in genome.Nodes)
            {
                writer.Write(node.Id);
                writer.Write((byte)node.Type);
            }

            writer.Write(genome.Connections.Count);
            foreach (var connection in genome.Connections)
            {
                writer.Write(connection.InNode);
                writer.Write(connection.OutNode);
                writer.Write(connection.Weight);
                writer.Write(connection.Enabled);
                writer.Write(connection.Innovation);
            }
        }

        private static NeatGenome ReadGenome(BinaryReader reader)
        {
            var inputCount = reader.ReadInt32();
            var outputCount = reader.ReadInt32();

            var nodeCount = reader.ReadInt32();
            var nodes = new List<NeatNodeGene>(nodeCount);
            for (var i = 0; i < nodeCount; i++)
            {
                var id = reader.ReadInt32();
                var type = (NeatNodeType)reader.ReadByte();
                nodes.Add(new NeatNodeGene(id, type));
            }

            var connectionCount = reader.ReadInt32();
            var connections = new List<NeatConnectionGene>(connectionCount);
            for (var i = 0; i < connectionCount; i++)
            {
                var inNode = reader.ReadInt32();
                var outNode = reader.ReadInt32();
                var weight = reader.ReadSingle();
                var enabled = reader.ReadBoolean();
                var innovation = reader.ReadInt32();
                connections.Add(new NeatConnectionGene(inNode, outNode, weight, enabled, innovation));
            }

            return new NeatGenome(inputCount, outputCount, nodes, connections);
        }
    }
}
