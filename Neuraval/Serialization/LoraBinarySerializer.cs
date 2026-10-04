using System;
using System.IO;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Neuraval.Core.Models;

namespace Neuraval.Core.Serialization
{
    public static class LoraBinarySerializer
    {
        private static readonly JsonSerializerOptions HeaderJsonOptions = new()
        {
            WriteIndented = false
        };

        public static void Save(string filePath, TransformerModelLoraState loraState, bool compress = true)
        {
            if (loraState == null) throw new ArgumentNullException(nameof(loraState));

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
                    LoraStateBinaryConverter.WriteTransformerModelLoraState(bodyWriter, loraState);
                }
                rawBody = bodyStream.ToArray();
            }

            var flags = LoraBinaryFormat.LoraFlags.None;
            byte[] bodyOnDisk;
            if (compress)
            {
                using var compressedStream = new MemoryStream();
                using (var gzip = new GZipStream(compressedStream, CompressionLevel.Optimal, leaveOpen: true))
                {
                    gzip.Write(rawBody, 0, rawBody.Length);
                }
                bodyOnDisk = compressedStream.ToArray();
                flags |= LoraBinaryFormat.LoraFlags.GZipCompressed;
            }
            else
            {
                bodyOnDisk = rawBody;
            }

            byte[] checksum = SHA256.HashData(bodyOnDisk);

            var rank = loraState.BlockStates.Count > 0 ? loraState.BlockStates[0].Query.Rank : 0;
            var alpha = loraState.BlockStates.Count > 0 ? loraState.BlockStates[0].Query.Alpha : 0f;

            var header = new LoraBinaryHeader
            {
                EmbeddingDim = loraState.EmbeddingDim,
                NumLayers = loraState.NumLayers,
                Rank = rank,
                Alpha = alpha,
                FreezeNonLoraWeights = loraState.FreezeNonLoraWeights,
                FormatVersion = LoraBinaryFormat.CurrentFormatVersion,
                Compressed = compress
            };
            byte[] headerJson = JsonSerializer.SerializeToUtf8Bytes(header, HeaderJsonOptions);

            var tempPath = filePath + ".tmp";
            using (var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write))
            using (var writer = new BinaryWriter(fileStream, Encoding.UTF8))
            {
                writer.Write(LoraBinaryFormat.MagicBytes);
                writer.Write(LoraBinaryFormat.CurrentFormatVersion);
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

        public static (TransformerModelLoraState LoraState, LoraBinaryHeader Header) Load(string filePath)
        {
            using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(fileStream, Encoding.UTF8);

            ushort formatVersion = ReadAndValidateFileHeader(reader, filePath);
            var flags = (LoraBinaryFormat.LoraFlags)reader.ReadByte();
            reader.ReadByte();

            int headerLength = reader.ReadInt32();
            var headerBytes = reader.ReadBytes(headerLength);
            var header = JsonSerializer.Deserialize<LoraBinaryHeader>(headerBytes)
                ?? throw new InvalidDataException($"No se pudo leer el encabezado del adaptador LoRA '{filePath}'.");
            header.FormatVersion = formatVersion;

            int bodyLength = reader.ReadInt32();
            var bodyOnDisk = reader.ReadBytes(bodyLength);

            var storedChecksum = reader.ReadBytes(LoraBinaryFormat.ChecksumLength);
            var actualChecksum = SHA256.HashData(bodyOnDisk);
            if (!storedChecksum.AsSpan().SequenceEqual(actualChecksum))
            {
                throw new InvalidDataException(
                    $"El archivo '{filePath}' está corrupto o incompleto: el checksum del cuerpo no coincide.");
            }

            byte[] rawBody = flags.HasFlag(LoraBinaryFormat.LoraFlags.GZipCompressed)
                ? Decompress(bodyOnDisk)
                : bodyOnDisk;

            using var bodyStream = new MemoryStream(rawBody);
            using var bodyReader = new BinaryReader(bodyStream, Encoding.UTF8);
            var loraState = LoraStateBinaryConverter.ReadTransformerModelLoraState(bodyReader);

            return (loraState, header);
        }

        public static bool IsNavloraFile(string filePath)
        {
            try
            {
                using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
                if (fileStream.Length < LoraBinaryFormat.MagicBytes.Length) return false;

                var magic = new byte[LoraBinaryFormat.MagicBytes.Length];
                int read = fileStream.Read(magic, 0, magic.Length);
                return read == magic.Length && magic.AsSpan().SequenceEqual(LoraBinaryFormat.MagicBytes);
            }
            catch (IOException)
            {
                return false;
            }
        }

        private static ushort ReadAndValidateFileHeader(BinaryReader reader, string filePath)
        {
            var magic = reader.ReadBytes(LoraBinaryFormat.MagicBytes.Length);
            if (!magic.AsSpan().SequenceEqual(LoraBinaryFormat.MagicBytes))
            {
                throw new InvalidDataException(
                    $"El archivo '{filePath}' no tiene la firma de un adaptador LoRA de Neuraval ({LoraBinaryFormat.FileExtension}).");
            }

            ushort formatVersion = reader.ReadUInt16();
            if (formatVersion > LoraBinaryFormat.CurrentFormatVersion)
            {
                throw new InvalidDataException(
                    $"'{filePath}' fue guardado con una versión de formato LoRA más nueva ({formatVersion}) " +
                    $"que la soportada por esta versión de Neuraval ({LoraBinaryFormat.CurrentFormatVersion}). " +
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
