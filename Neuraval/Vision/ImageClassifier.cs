using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using Neuraval.Core.Models;

namespace Neuraval.Core.Vision;

public sealed class ImageClassifier
{
    private static readonly byte[] Magic = "NVIMG001"u8.ToArray();
    private readonly FeedForwardNetwork _network;

    public int ImageSize { get; }
    public IReadOnlyList<string> Labels { get; }

    private ImageClassifier(int imageSize, IReadOnlyList<string> labels, FeedForwardNetwork network)
    {
        ImageSize = imageSize;
        Labels = labels;
        _network = network;
    }

    public static ImageClassifier Train(
        string datasetDirectory,
        int imageSize = 28,
        int epochs = 10,
        int hiddenSize = 128,
        float learningRate = 0.001f,
        int seed = 42,
        Action<string>? report = null)
    {
        if (imageSize is < 4 or > 256) throw new ArgumentOutOfRangeException(nameof(imageSize));
        if (epochs <= 0) throw new ArgumentOutOfRangeException(nameof(epochs));
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (!float.IsFinite(learningRate) || learningRate <= 0) throw new ArgumentOutOfRangeException(nameof(learningRate));

        var classes = LoadLabeledImages(datasetDirectory, imageSize);
        if (classes.Count < 2) throw new InvalidDataException("El dataset debe tener al menos dos carpetas de clases.");
        if (classes.Any(c => c.Images.Count == 0)) throw new InvalidDataException("Cada carpeta de clase debe contener imágenes PNG o BMP compatibles.");

        var labels = classes.Select(c => c.Label).ToArray();
        var samples = new List<(float[] Pixels, int Label)>();
        for (int label = 0; label < classes.Count; label++)
            samples.AddRange(classes[label].Images.Select(pixels => (pixels, label)));

        var network = new FeedForwardNetwork(imageSize * imageSize, hiddenSize, seed, classes.Count);
        var classifier = new ImageClassifier(imageSize, labels, network);
        var random = new Random(seed);
        var order = Enumerable.Range(0, samples.Count).ToArray();

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            Shuffle(order, random);
            double loss = 0;
            int correct = 0;
            foreach (int sampleIndex in order)
            {
                var sample = samples[sampleIndex];
                var input = ToMatrix(sample.Pixels);
                var logits = network.Forward(input);
                var probabilities = Softmax(logits);
                loss -= Math.Log(Math.Max(probabilities[sample.Label], 1e-12));
                if (ArgMax(probabilities) == sample.Label) correct++;

                probabilities[sample.Label] -= 1f;
                network.ZeroGradients();
                network.Backward(ToMatrix(probabilities), learningRate);
                network.AverageGradients(1);
                network.UpdateWeights(learningRate);
            }

            report?.Invoke($"Época {epoch}/{epochs}: pérdida {loss / samples.Count:F4}; precisión de entrenamiento {100.0 * correct / samples.Count:F1}%");
        }

        return classifier;
    }

    public (string Label, float Confidence) Predict(string imagePath)
    {
        var logits = _network.Forward(ToMatrix(ImageFile.ReadGrayscale(imagePath, ImageSize)));
        var probabilities = Softmax(logits);
        int best = ArgMax(probabilities);
        return (Labels[best], probabilities[best]);
    }

    public (int Correct, int Total, double Accuracy) Evaluate(string datasetDirectory)
    {
        var classes = LoadLabeledImages(datasetDirectory, ImageSize);
        if (classes.Any(c => !Labels.Contains(c.Label, StringComparer.Ordinal)))
            throw new InvalidDataException("El dataset de prueba contiene carpetas de clases que no existen en el modelo.");

        int correct = 0;
        int total = 0;
        foreach (var imageClass in classes)
        foreach (var pixels in imageClass.Images)
        {
            int expected = IndexOfLabel(imageClass.Label);
            var probabilities = Softmax(_network.Forward(ToMatrix(pixels)));
            if (ArgMax(probabilities) == expected) correct++;
            total++;
        }

        if (total == 0) throw new InvalidDataException("No se encontraron imágenes PNG o BMP en el dataset de prueba.");
        return (correct, total, (double)correct / total);
    }

    public void Save(string path)
    {
        string fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        string temporaryPath = fullPath + ".tmp";
        try
        {
            using (var stream = new FileStream(temporaryPath, FileMode.Create, FileAccess.Write, FileShare.None))
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false))
            {
                writer.Write(Magic);
                writer.Write(ImageSize);
                writer.Write(_network.HiddenDim);
                writer.Write(Labels.Count);
                foreach (string label in Labels) writer.Write(label);

                var state = _network.SaveState();
                WriteFloats(writer, state.Weights1);
                WriteFloats(writer, state.Bias1);
                WriteFloats(writer, state.Weights2);
                WriteFloats(writer, state.Bias2);
            }

            File.Move(temporaryPath, fullPath, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporaryPath)) File.Delete(temporaryPath);
        }
    }

    public static ImageClassifier Load(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false);
        if (!reader.ReadBytes(Magic.Length).SequenceEqual(Magic)) throw new InvalidDataException("El archivo no es un modelo de imágenes Neuraval válido.");

        int imageSize = reader.ReadInt32();
        int hiddenSize = reader.ReadInt32();
        int labelCount = reader.ReadInt32();
        if (imageSize is < 4 or > 256 || hiddenSize <= 0 || labelCount is < 2 or > 10000)
            throw new InvalidDataException("La arquitectura del modelo contiene dimensiones inválidas.");

        var labels = new string[labelCount];
        for (int i = 0; i < labelCount; i++) labels[i] = reader.ReadString();
        if (labels.Any(string.IsNullOrWhiteSpace) || labels.Distinct(StringComparer.Ordinal).Count() != labels.Length)
            throw new InvalidDataException("El modelo contiene etiquetas vacías o repetidas.");

        int inputSize = checked(imageSize * imageSize);
        var state = new FeedForwardNetworkState
        {
            EmbeddingDim = inputSize,
            HiddenDim = hiddenSize,
            Weights1 = ReadFloats(reader, checked(inputSize * hiddenSize)),
            Bias1 = ReadFloats(reader, hiddenSize),
            Weights2 = ReadFloats(reader, checked(hiddenSize * labelCount)),
            Bias2 = ReadFloats(reader, labelCount)
        };
        if (stream.Position != stream.Length) throw new InvalidDataException("El archivo del modelo contiene datos adicionales no reconocidos.");

        return new ImageClassifier(imageSize, labels, FeedForwardNetwork.LoadState(state));
    }

    private int IndexOfLabel(string label)
    {
        for (int i = 0; i < Labels.Count; i++)
            if (string.Equals(Labels[i], label, StringComparison.Ordinal)) return i;
        throw new InvalidDataException($"La clase '{label}' no existe en el modelo.");
    }

    private static List<ImageClass> LoadLabeledImages(string directory, int imageSize)
    {
        if (!Directory.Exists(directory)) throw new DirectoryNotFoundException($"No existe el directorio de imágenes: {directory}");
        var result = new List<ImageClass>();
        foreach (string classDirectory in Directory.GetDirectories(directory).OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase))
        {
            var imageClass = new ImageClass(Path.GetFileName(classDirectory));
            foreach (string file in Directory.EnumerateFiles(classDirectory, "*", SearchOption.AllDirectories)
                         .Where(ImageFile.IsSupported).OrderBy(p => p, StringComparer.OrdinalIgnoreCase))
                imageClass.Images.Add(ImageFile.ReadGrayscale(file, imageSize));
            result.Add(imageClass);
        }
        return result;
    }

    private static float[,] ToMatrix(float[] input)
    {
        var result = new float[1, input.Length];
        for (int i = 0; i < input.Length; i++) result[0, i] = input[i];
        return result;
    }

    private static float[] Softmax(float[,] logits)
    {
        int count = logits.GetLength(1);
        var result = new float[count];
        float max = logits[0, 0];
        for (int i = 1; i < count; i++) max = Math.Max(max, logits[0, i]);
        double sum = 0;
        for (int i = 0; i < count; i++) sum += result[i] = MathF.Exp(logits[0, i] - max);
        for (int i = 0; i < count; i++) result[i] = (float)(result[i] / sum);
        return result;
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    private static void Shuffle(int[] values, Random random)
    {
        for (int i = values.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    private static void WriteFloats(BinaryWriter writer, float[] values)
    {
        writer.Write(values.Length);
        foreach (float value in values) writer.Write(value);
    }

    private static float[] ReadFloats(BinaryReader reader, int expectedLength)
    {
        int length = reader.ReadInt32();
        if (length != expectedLength) throw new InvalidDataException("El tamaño de pesos del modelo no coincide con su arquitectura.");
        var values = new float[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = reader.ReadSingle();
            if (!float.IsFinite(values[i])) throw new InvalidDataException("El modelo contiene pesos no finitos.");
        }
        return values;
    }

    private sealed class ImageClass(string label)
    {
        public string Label { get; } = label;
        public List<float[]> Images { get; } = new();
    }
}

internal static class ImageFile
{
    private static readonly byte[] PngSignature = { 137, 80, 78, 71, 13, 10, 26, 10 };

    public static bool IsSupported(string path) =>
        string.Equals(Path.GetExtension(path), ".bmp", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(Path.GetExtension(path), ".png", StringComparison.OrdinalIgnoreCase);

    public static float[] ReadGrayscale(string path, int targetSize)
    {
        byte[] file = File.ReadAllBytes(path);
        var image = file.AsSpan().StartsWith(PngSignature) ? DecodePng(file) : DecodeBmp(file);
        var result = new float[targetSize * targetSize];
        for (int y = 0; y < targetSize; y++)
        {
            int sourceY = Math.Min(image.Height - 1, y * image.Height / targetSize);
            for (int x = 0; x < targetSize; x++)
            {
                int sourceX = Math.Min(image.Width - 1, x * image.Width / targetSize);
                result[y * targetSize + x] = image.Pixels[sourceY * image.Width + sourceX] / 255f;
            }
        }
        return result;
    }

    private static GrayImage DecodeBmp(byte[] data)
    {
        if (data.Length < 54 || data[0] != 'B' || data[1] != 'M') throw new InvalidDataException("Formato no compatible; se aceptan imágenes PNG o BMP sin compresión.");
        int offset = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(10, 4));
        int headerSize = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(14, 4));
        int width = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(18, 4));
        int signedHeight = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(22, 4));
        ushort planes = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(26, 2));
        ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(28, 2));
        uint compression = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(30, 4));
        if (headerSize < 40 || width <= 0 || signedHeight == 0 || signedHeight == int.MinValue || planes != 1 || bits is not (24 or 32) || compression != 0)
            throw new InvalidDataException("BMP incompatible: usa 24/32 bits, sin compresión y dimensiones positivas.");
        int height = Math.Abs(signedHeight);
        ValidateImageSize(width, height);
        int bytesPerPixel = bits / 8;
        int stride = checked(((width * bits + 31) / 32) * 4);
        if (offset < 0 || (long)offset + (long)stride * height > data.Length) throw new InvalidDataException("El BMP está incompleto o dañado.");
        var pixels = new byte[width * height];
        for (int y = 0; y < height; y++)
        {
            int sourceY = signedHeight > 0 ? height - 1 - y : y;
            int row = offset + sourceY * stride;
            for (int x = 0; x < width; x++)
            {
                int p = row + x * bytesPerPixel;
                pixels[y * width + x] = ToGray(data[p + 2], data[p + 1], data[p]);
            }
        }
        return new GrayImage(width, height, pixels);
    }

    private static GrayImage DecodePng(byte[] data)
    {
        int offset = PngSignature.Length;
        int width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
        using var compressed = new MemoryStream();
        while (offset + 12 <= data.Length)
        {
            int length = BinaryPrimitives.ReadInt32BigEndian(data.AsSpan(offset, 4));
            if (length < 0 || (long)offset + 12 + length > data.Length) throw new InvalidDataException("PNG inválido o incompleto.");
            string type = Encoding.ASCII.GetString(data, offset + 4, 4);
            var payload = data.AsSpan(offset + 8, length);
            if (type == "IHDR")
            {
                if (length != 13) throw new InvalidDataException("Cabecera PNG inválida.");
                width = BinaryPrimitives.ReadInt32BigEndian(payload[..4]);
                height = BinaryPrimitives.ReadInt32BigEndian(payload.Slice(4, 4));
                bitDepth = payload[8]; colorType = payload[9]; interlace = payload[12];
            }
            else if (type == "IDAT") compressed.Write(payload);
            else if (type == "IEND") break;
            offset += 12 + length;
        }

        ValidateImageSize(width, height);
        int channels = colorType switch { 0 => 1, 2 => 3, 4 => 2, 6 => 4, _ => 0 };
        if (bitDepth != 8 || channels == 0 || interlace != 0)
            throw new InvalidDataException("PNG incompatible: se acepta PNG de 8 bits no entrelazado en escala de grises o RGB/RGBA.");

        int stride = checked(width * channels);
        byte[] raw = new byte[checked((stride + 1) * height)];
        compressed.Position = 0;
        using (var zlib = new ZLibStream(compressed, CompressionMode.Decompress, leaveOpen: true))
        {
            int read = 0;
            while (read < raw.Length)
            {
                int count = zlib.Read(raw, read, raw.Length - read);
                if (count == 0) throw new InvalidDataException("PNG truncado: faltan datos de píxeles.");
                read += count;
            }
        }

        var decoded = new byte[stride * height];
        int rawOffset = 0;
        for (int y = 0; y < height; y++)
        {
            byte filter = raw[rawOffset++];
            int row = y * stride;
            for (int x = 0; x < stride; x++)
            {
                int left = x >= channels ? decoded[row + x - channels] : 0;
                int up = y > 0 ? decoded[row + x - stride] : 0;
                int upperLeft = y > 0 && x >= channels ? decoded[row + x - stride - channels] : 0;
                int predictor = filter switch
                {
                    0 => 0,
                    1 => left,
                    2 => up,
                    3 => (left + up) / 2,
                    4 => Paeth(left, up, upperLeft),
                    _ => throw new InvalidDataException($"Filtro PNG desconocido: {filter}.")
                };
                decoded[row + x] = unchecked((byte)(raw[rawOffset++] + predictor));
            }
        }

        var pixels = new byte[width * height];
        for (int i = 0; i < pixels.Length; i++)
        {
            int p = i * channels;
            pixels[i] = colorType switch
            {
                0 => decoded[p],
                2 => ToGray(decoded[p], decoded[p + 1], decoded[p + 2]),
                4 => CompositeGray(decoded[p], decoded[p + 1]),
                6 => CompositeGray(ToGray(decoded[p], decoded[p + 1], decoded[p + 2]), decoded[p + 3]),
                _ => 0
            };
        }
        return new GrayImage(width, height, pixels);
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }

    private static byte ToGray(byte red, byte green, byte blue) => (byte)((77 * red + 150 * green + 29 * blue + 128) >> 8);
    private static byte CompositeGray(byte gray, byte alpha) => (byte)((gray * alpha + 255 * (255 - alpha) + 127) / 255);

    private static void ValidateImageSize(int width, int height)
    {
        if (width <= 0 || height <= 0 || (long)width * height > 64_000_000)
            throw new InvalidDataException("Dimensiones de imagen inválidas o demasiado grandes.");
    }

    private sealed record GrayImage(int Width, int Height, byte[] Pixels);
}
