using System.IO;
using System.Text.Json;
using Neuraval.Evolution.Serialization;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioDatasetHeader
    {
        public const string KindValue = "imitation-dataset";
        public string Kind { get; set; } = KindValue;
        public int FormatVersion { get; set; }
        public int InputCount { get; set; }
        public int OutputCount { get; set; }
        public long RecordCount { get; set; }
        public DateTime SavedAtUtc { get; set; }
    }

    public sealed class MarioDatasetRecorder : IDisposable
    {
        public const int CurrentFormatVersion = 1;

        private static readonly JsonSerializerOptions HeaderJsonOptions = new() { WriteIndented = false };

        private readonly string _finalPath;
        private readonly string _tempPath;
        private FileStream? _stream;
        private BinaryWriter? _writer;
        private long _recordCount;

        public MarioDatasetRecorder(string filePath)
        {
            _finalPath = filePath;
            _tempPath = filePath + ".tmp";

            var fullPath = Path.GetFullPath(filePath);
            var directory = Path.GetDirectoryName(fullPath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            _stream = new FileStream(_tempPath, FileMode.Create, FileAccess.Write);
            _writer = new BinaryWriter(_stream);
        }

        public void Append(SnesState state, SnesButton action, float reward, bool done)
        {
            var input = MarioStateEncoder.Encode(state);
            _writer!.Write(state.Frame);
            _writer.Write(state.LevelIndex);
            foreach (var value in input)
            {
                _writer.Write(value);
            }

            _writer.Write((ushort)action);
            _writer.Write(reward);
            _writer.Write(done);
            _recordCount++;
        }

        public void Complete()
        {
            if (_writer == null)
            {
                return;
            }

            _writer.Flush();
            _stream!.Flush(flushToDisk: true);

            _stream.Seek(0, SeekOrigin.Begin);
            byte[] body;
            using (var bodyStream = new MemoryStream())
            {
                _stream.CopyTo(bodyStream);
                body = bodyStream.ToArray();
            }

            _writer.Dispose();
            _stream.Dispose();
            _writer = null;
            _stream = null;

            var header = new MarioDatasetHeader
            {
                FormatVersion = CurrentFormatVersion,
                InputCount = MarioAgent.InputCount,
                OutputCount = MarioAgent.OutputCount,
                RecordCount = _recordCount,
                SavedAtUtc = DateTime.UtcNow
            };

            NavmBinarySerializer.Save(_finalPath, JsonSerializer.Serialize(header, HeaderJsonOptions), body);
            File.Delete(_tempPath);

            Console.WriteLine($"Dataset guardado: {_recordCount} muestras en {Path.GetFullPath(_finalPath)}.");
        }

        public void Dispose()
        {
            if (_writer == null)
            {
                return;
            }

            _writer.Dispose();
            _stream!.Dispose();
            _writer = null;
            _stream = null;

            if (File.Exists(_tempPath))
            {
                File.Delete(_tempPath);
            }

            Console.WriteLine("Captura abortada: el dataset no se guardo.");
        }
    }

    public sealed record MarioDatasetSample(int Frame, int LevelIndex, float[] Input, SnesButton ActionMask, float Reward, bool Done);

    public static class MarioDatasetLoader
    {
        public static MarioDataset? Load(string filePath)
        {
            var content = NavmBinarySerializer.Load(filePath);
            if (content == null)
            {
                return null;
            }

            MarioDatasetHeader? header;
            try
            {
                header = JsonSerializer.Deserialize<MarioDatasetHeader>(content.HeaderJson);
            }
            catch (JsonException)
            {
                return null;
            }

            if (header == null || header.Kind != MarioDatasetHeader.KindValue)
            {
                Console.WriteLine($"El archivo {filePath} no es un dataset de imitacion valido.");
                return null;
            }

            if (header.InputCount != MarioAgent.InputCount || header.OutputCount != MarioAgent.OutputCount)
            {
                Console.WriteLine(
                    $"El dataset {filePath} es incompatible con la red actual: " +
                    $"fue generado con {header.InputCount} entradas / {header.OutputCount} salidas, " +
                    $"pero el codigo actual usa {MarioAgent.InputCount} entradas / {MarioAgent.OutputCount} salidas. " +
                    "Recondena el dataset con el codigo actual.");
                return null;
            }

            var dataset = new MarioDataset(header.InputCount, header.OutputCount);

            using var stream = new MemoryStream(content.Body);
            using var reader = new BinaryReader(stream);
            for (var i = 0; i < header.RecordCount; i++)
            {
                var frame = reader.ReadInt32();
                var levelIndex = reader.ReadInt32();

                var input = new float[header.InputCount];
                for (var j = 0; j < header.InputCount; j++)
                {
                    input[j] = reader.ReadSingle();
                }

                var actionMask = (SnesButton)reader.ReadUInt16();
                var reward = reader.ReadSingle();
                var done = reader.ReadByte() != 0;

                dataset.Samples.Add(new MarioDatasetSample(frame, levelIndex, input, actionMask, reward, done));
            }

            return dataset;
        }
    }

    public sealed class MarioDataset
    {
        public int InputCount { get; }
        public int OutputCount { get; }
        public List<MarioDatasetSample> Samples { get; } = new List<MarioDatasetSample>();

        public MarioDataset(int inputCount, int outputCount)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
        }
    }
}