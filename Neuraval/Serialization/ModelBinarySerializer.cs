using System;
using System.IO;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Neuraval.Core.Models;

namespace Neuraval.Core.Serialization
{
    public static class ModelBinarySerializer
    {
        private static readonly JsonSerializerOptions HeaderJsonOptions = new()
        {
            WriteIndented = false
        };

        public static void Save(string filePath, TransformerModelState modelState, ModelBinaryHeader header, bool compress = true)
        {
            if (modelState == null) throw new ArgumentNullException(nameof(modelState));
            if (header == null) throw new ArgumentNullException(nameof(header));

            var directory = Path.GetDirectoryName(Path.GetFullPath(filePath));
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            byte[] rawBody;
            using (var bodyStream = new MemoryStream())
            {
                using (var bodyWriter = new BinaryWriter(bodyStream, Encoding.UTF8, leaveOpen: true))
                {
                    ModelStateBinaryConverter.WriteTransformerModelState(bodyWriter, modelState);
                }
                rawBody = bodyStream.ToArray();
            }

            var flags = ModelBinaryFormat.ModelFlags.None;
            byte[] bodyOnDisk;
            if (compress)
            {
                using var compressedStream = new MemoryStream();
                using (var gzip = new GZipStream(compressedStream, CompressionLevel.Optimal, leaveOpen: true))
                {
                    gzip.Write(rawBody, 0, rawBody.Length);
                }
                bodyOnDisk = compressedStream.ToArray();
                flags |= ModelBinaryFormat.ModelFlags.GZipCompressed;
            }
            else
            {
                bodyOnDisk = rawBody;
            }

            byte[] checksum = SHA256.HashData(bodyOnDisk);

            header.FormatVersion = ModelBinaryFormat.CurrentFormatVersion;
            header.Compressed = compress;
            byte[] headerJson = JsonSerializer.SerializeToUtf8Bytes(header, HeaderJsonOptions);

            var tempPath = filePath + ".tmp";
            using (var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write))
            using (var writer = new BinaryWriter(fileStream, Encoding.UTF8))
            {
                writer.Write(ModelBinaryFormat.MagicBytes);
                writer.Write(ModelBinaryFormat.CurrentFormatVersion);
                writer.Write((byte)flags);
                writer.Write((byte)0);
                writer.Write(headerJson.Length);
                writer.Write(headerJson);
                writer.Write(bodyOnDisk.Length);
                writer.Write(bodyOnDisk);
                writer.Write(checksum);
            }

            File.Move(tempPath, filePath, overwrite: true);
        }

        public static void SaveQuantized(string filePath, TransformerModelState modelState, ModelBinaryHeader header, bool compress = true)
        {
            if (modelState == null) throw new ArgumentNullException(nameof(modelState));
            if (header == null) throw new ArgumentNullException(nameof(header));

            var directory = Path.GetDirectoryName(Path.GetFullPath(filePath));
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            byte[] rawBody;
            using (var bodyStream = new MemoryStream())
            {
                using (var bodyWriter = new BinaryWriter(bodyStream, Encoding.UTF8, leaveOpen: true))
                {
                    QuantizedModelStateBinaryConverter.WriteTransformerModelStateQuantized(bodyWriter, modelState);
                }
                rawBody = bodyStream.ToArray();
            }

            var flags = ModelBinaryFormat.ModelFlags.Int8QuantizedWeights;
            byte[] bodyOnDisk;
            if (compress)
            {
                using var compressedStream = new MemoryStream();
                using (var gzip = new GZipStream(compressedStream, CompressionLevel.Optimal, leaveOpen: true))
                {
                    gzip.Write(rawBody, 0, rawBody.Length);
                }
                bodyOnDisk = compressedStream.ToArray();
                flags |= ModelBinaryFormat.ModelFlags.GZipCompressed;
            }
            else
            {
                bodyOnDisk = rawBody;
            }

            byte[] checksum = SHA256.HashData(bodyOnDisk);

            header.FormatVersion = ModelBinaryFormat.CurrentFormatVersion;
            header.Compressed = compress;
            header.Quantized = true;
            header.QuantizationScheme = "int8-simetrico-por-fila";
            byte[] headerJson = JsonSerializer.SerializeToUtf8Bytes(header, HeaderJsonOptions);

            var tempPath = filePath + ".tmp";
            using (var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write))
            using (var writer = new BinaryWriter(fileStream, Encoding.UTF8))
            {
                writer.Write(ModelBinaryFormat.MagicBytes);
                writer.Write(ModelBinaryFormat.CurrentFormatVersion);
                writer.Write((byte)flags);
                writer.Write((byte)0);
                writer.Write(headerJson.Length);
                writer.Write(headerJson);
                writer.Write(bodyOnDisk.Length);
                writer.Write(bodyOnDisk);
                writer.Write(checksum);
            }

            File.Move(tempPath, filePath, overwrite: true);
        }

        public static (TransformerModelState ModelState, ModelBinaryHeader Header) Load(string filePath)
        {
            using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(fileStream, Encoding.UTF8);

            ushort formatVersion = ReadAndValidateFileHeader(reader, filePath);
            var flags = (ModelBinaryFormat.ModelFlags)reader.ReadByte();
            reader.ReadByte();

            int headerLength = reader.ReadInt32();
            var headerBytes = reader.ReadBytes(headerLength);
            var header = JsonSerializer.Deserialize<ModelBinaryHeader>(headerBytes)
                ?? throw new InvalidDataException($"No se pudo leer el encabezado del modelo '{filePath}'.");
            header.FormatVersion = formatVersion;

            int bodyLength = reader.ReadInt32();
            var bodyOnDisk = reader.ReadBytes(bodyLength);

            var storedChecksum = reader.ReadBytes(ModelBinaryFormat.ChecksumLength);
            var actualChecksum = SHA256.HashData(bodyOnDisk);
            if (!storedChecksum.AsSpan().SequenceEqual(actualChecksum))
            {
                throw new InvalidDataException(
                    $"El archivo '{filePath}' está corrupto o incompleto: el checksum del cuerpo no coincide.");
            }

            byte[] rawBody = flags.HasFlag(ModelBinaryFormat.ModelFlags.GZipCompressed)
                ? Decompress(bodyOnDisk)
                : bodyOnDisk;

            bool isQuantized = flags.HasFlag(ModelBinaryFormat.ModelFlags.Int8QuantizedWeights);
            header.Quantized = isQuantized;

            using var bodyStream = new MemoryStream(rawBody);
            using var bodyReader = new BinaryReader(bodyStream, Encoding.UTF8);
            var modelState = isQuantized
                ? QuantizedModelStateBinaryConverter.ReadTransformerModelStateQuantized(bodyReader)
                : ModelStateBinaryConverter.ReadTransformerModelState(bodyReader, formatVersion);

            return (modelState, header);
        }

        public static ModelBinaryHeader ReadHeaderOnly(string filePath)
        {
            using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(fileStream, Encoding.UTF8);

            ushort formatVersion = ReadAndValidateFileHeader(reader, filePath);
            reader.ReadByte();
            reader.ReadByte();

            int headerLength = reader.ReadInt32();
            var headerBytes = reader.ReadBytes(headerLength);
            var header = JsonSerializer.Deserialize<ModelBinaryHeader>(headerBytes)
                ?? throw new InvalidDataException($"No se pudo leer el encabezado del modelo '{filePath}'.");
            header.FormatVersion = formatVersion;
            return header;
        }

        public static bool IsNavmFile(string filePath)
        {
            try
            {
                using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
                if (fileStream.Length < ModelBinaryFormat.MagicBytes.Length) return false;

                var magic = new byte[ModelBinaryFormat.MagicBytes.Length];
                int read = fileStream.Read(magic, 0, magic.Length);
                return read == magic.Length && magic.AsSpan().SequenceEqual(ModelBinaryFormat.MagicBytes);
            }
            catch (IOException)
            {
                return false;
            }
        }

        private static ushort ReadAndValidateFileHeader(BinaryReader reader, string filePath)
        {
            var magic = reader.ReadBytes(ModelBinaryFormat.MagicBytes.Length);
            if (!magic.AsSpan().SequenceEqual(ModelBinaryFormat.MagicBytes))
            {
                throw new InvalidDataException(
                    $"El archivo '{filePath}' no tiene la firma de un modelo Neuraval ({ModelBinaryFormat.FileExtension}).");
            }

            ushort formatVersion = reader.ReadUInt16();
            if (formatVersion > ModelBinaryFormat.CurrentFormatVersion)
            {
                throw new InvalidDataException(
                    $"'{filePath}' fue guardado con una versión de formato más nueva ({formatVersion}) " +
                    $"que la soportada por esta versión de Neuraval ({ModelBinaryFormat.CurrentFormatVersion}). " +
                    "Actualizá la aplicación para poder cargarlo.");
            }

            return formatVersion;
        }

        private static byte[] Decompress(byte[] compressed)
        {
            using var compressedStream = new MemoryStream(compressed);
            using var gzip = new GZipStream(compressedStream, CompressionMode.Decompress);
            using var decompressed = new MemoryStream();
            gzip.CopyTo(decompressed);
            return decompressed.ToArray();
        }
    }
}
