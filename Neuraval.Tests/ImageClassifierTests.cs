using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using Neuraval.Core.Vision;
using Xunit;

namespace Neuraval.Tests;

public class ImageClassifierTests
{
    [Fact]
    public void Train_Save_LoadAndEvaluate_ClassifiesSimpleBitmapPatterns()
    {
        string root = Path.Combine(Path.GetTempPath(), "neuraval-vision-" + Guid.NewGuid().ToString("N"));
        string training = Path.Combine(root, "train");
        string testing = Path.Combine(root, "test");
        Directory.CreateDirectory(Path.Combine(training, "vertical"));
        Directory.CreateDirectory(Path.Combine(training, "horizontal"));
        Directory.CreateDirectory(Path.Combine(testing, "vertical"));
        Directory.CreateDirectory(Path.Combine(testing, "horizontal"));

        try
        {
            WriteBmp(Path.Combine(training, "vertical", "one.bmp"), vertical: true);
            WriteBmp(Path.Combine(training, "horizontal", "one.bmp"), vertical: false);
            WriteBmp(Path.Combine(testing, "vertical", "one.bmp"), vertical: true);
            WriteBmp(Path.Combine(testing, "horizontal", "one.bmp"), vertical: false);
            WritePng(Path.Combine(testing, "vertical", "png.png"), vertical: true);

            var classifier = ImageClassifier.Train(training, imageSize: 4, epochs: 40, hiddenSize: 12, learningRate: 0.01f);
            string modelPath = Path.Combine(root, "classifier.nvimg");
            classifier.Save(modelPath);

            var loaded = ImageClassifier.Load(modelPath);
            var report = loaded.Evaluate(testing);
            var prediction = loaded.Predict(Path.Combine(testing, "vertical", "one.bmp"));

            Assert.Equal(2, loaded.Labels.Count);
            Assert.Equal(3, report.Total);
            Assert.Equal(3, report.Correct);
            Assert.Equal("vertical", prediction.Label);
            Assert.True(prediction.Confidence > 0.5f);
            Assert.Equal("vertical", loaded.Predict(Path.Combine(testing, "vertical", "png.png")).Label);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    private static void WriteBmp(string path, bool vertical)
    {
        const int width = 4, height = 4, stride = 12;
        byte[] bmp = new byte[54 + stride * height];
        bmp[0] = (byte)'B'; bmp[1] = (byte)'M';
        BinaryPrimitives.WriteInt32LittleEndian(bmp.AsSpan(2, 4), bmp.Length);
        BinaryPrimitives.WriteInt32LittleEndian(bmp.AsSpan(10, 4), 54);
        BinaryPrimitives.WriteInt32LittleEndian(bmp.AsSpan(14, 4), 40);
        BinaryPrimitives.WriteInt32LittleEndian(bmp.AsSpan(18, 4), width);
        BinaryPrimitives.WriteInt32LittleEndian(bmp.AsSpan(22, 4), height);
        BinaryPrimitives.WriteUInt16LittleEndian(bmp.AsSpan(26, 2), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(bmp.AsSpan(28, 2), 24);
        BinaryPrimitives.WriteInt32LittleEndian(bmp.AsSpan(34, 4), stride * height);

        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            bool ink = vertical ? x is 1 or 2 : y is 1 or 2;
            byte value = ink ? (byte)0 : (byte)255;
            int offset = 54 + (height - 1 - y) * stride + x * 3;
            bmp[offset] = value; bmp[offset + 1] = value; bmp[offset + 2] = value;
        }

        File.WriteAllBytes(path, bmp);
    }

    private static void WritePng(string path, bool vertical)
    {
        const int width = 4, height = 4, channels = 3;
        byte[] scanlines = new byte[(width * channels + 1) * height];
        for (int y = 0; y < height; y++)
        {
            int row = y * (width * channels + 1);
            scanlines[row] = 0;
            for (int x = 0; x < width; x++)
            {
                byte value = (vertical ? x is 1 or 2 : y is 1 or 2) ? (byte)0 : (byte)255;
                int pixel = row + 1 + x * channels;
                scanlines[pixel] = value;
                scanlines[pixel + 1] = value;
                scanlines[pixel + 2] = value;
            }
        }

        using var compressed = new MemoryStream();
        using (var zlib = new ZLibStream(compressed, CompressionLevel.Optimal, leaveOpen: true))
            zlib.Write(scanlines);

        using var png = new MemoryStream();
        png.Write(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });
        var header = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(0, 4), width);
        BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4, 4), height);
        header[8] = 8;
        header[9] = 2;
        WritePngChunk(png, "IHDR", header);
        WritePngChunk(png, "IDAT", compressed.ToArray());
        WritePngChunk(png, "IEND", Array.Empty<byte>());
        File.WriteAllBytes(path, png.ToArray());
    }

    private static void WritePngChunk(Stream stream, string type, byte[] data)
    {
        Span<byte> length = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(length, data.Length);
        stream.Write(length);
        byte[] typeBytes = Encoding.ASCII.GetBytes(type);
        stream.Write(typeBytes);
        stream.Write(data);
        stream.Write(new byte[4]); // The decoder ignores CRC; chunk structure remains valid for this fixture.
    }
}
