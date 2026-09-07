using System;
using System.IO;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Neuraval.Core.Models;

namespace Neuraval.Core.Serialization
{
    /// <summary>
    /// Guarda y carga modelos <see cref="TransformerModel"/> completos (pesos +
    /// estado de los optimizadores) usando el formato binario propietario
    /// <c>.ncbm</c> descrito en <see cref="ModelBinaryFormat"/>.
    /// </summary>
    public static class ModelBinarySerializer
    {
        private static readonly JsonSerializerOptions HeaderJsonOptions = new()
        {
            WriteIndented = false
        };

        /// <summary>
        /// Serializa <paramref name="modelState"/> y lo escribe en
        /// <paramref name="filePath"/> con el formato <c>.ncbm</c>.
        /// </summary>
        /// <param name="compress">
        /// Si es <c>true</c> (default), el cuerpo binario se comprime con GZip.
        /// Los pesos de una red entrenada suelen comprimir bien porque tienen
        /// muchos valores pequeños y repetitivos, lo que reduce bastante el
        /// tamaño en disco a costa de un poco más de CPU al guardar/cargar.
        /// </param>
        public static void Save(string filePath, TransformerModelState modelState, ModelBinaryHeader header, bool compress = true)
        {
            if (modelState == null) throw new ArgumentNullException(nameof(modelState));
            if (header == null) throw new ArgumentNullException(nameof(header));

            var directory = Path.GetDirectoryName(Path.GetFullPath(filePath));
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // 1) Serializar los pesos + optimizadores al layout binario propio.
            byte[] rawBody;
            using (var bodyStream = new MemoryStream())
            {
                using (var bodyWriter = new BinaryWriter(bodyStream, Encoding.UTF8, leaveOpen: true))
                {
                    ModelStateBinaryConverter.WriteTransformerModelState(bodyWriter, modelState);
                }
                rawBody = bodyStream.ToArray();
            }

            // 2) Comprimir opcionalmente el cuerpo.
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

            // 3) Checksum de integridad sobre los bytes tal cual quedan en disco.
            byte[] checksum = SHA256.HashData(bodyOnDisk);

            // 4) Encabezado auto-descriptivo, versionado de forma independiente al Body.
            header.FormatVersion = ModelBinaryFormat.CurrentFormatVersion;
            header.Compressed = compress;
            byte[] headerJson = JsonSerializer.SerializeToUtf8Bytes(header, HeaderJsonOptions);

            // Escribir a un archivo temporal y luego mover: evita dejar un .ncbm
            // corrupto/a medio escribir si el proceso se interrumpe justo al guardar
            // un checkpoint (buena práctica para archivos que se sobrescriben seguido).
            var tempPath = filePath + ".tmp";
            using (var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write))
            using (var writer = new BinaryWriter(fileStream, Encoding.UTF8))
            {
                writer.Write(ModelBinaryFormat.MagicBytes);
                writer.Write(ModelBinaryFormat.CurrentFormatVersion);
                writer.Write((byte)flags);
                writer.Write((byte)0); // reservado
                writer.Write(headerJson.Length);
                writer.Write(headerJson);
                writer.Write(bodyOnDisk.Length);
                writer.Write(bodyOnDisk);
                writer.Write(checksum);
            }

            File.Move(tempPath, filePath, overwrite: true);
        }

        /// <summary>
        /// Carga un archivo <c>.ncbm</c> completo: valida la firma, la versión
        /// de formato y el checksum, y devuelve tanto los pesos como el
        /// encabezado con los metadatos del modelo.
        /// </summary>
        public static (TransformerModelState ModelState, ModelBinaryHeader Header) Load(string filePath)
        {
            using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(fileStream, Encoding.UTF8);

            ushort formatVersion = ReadAndValidateFileHeader(reader, filePath);
            var flags = (ModelBinaryFormat.ModelFlags)reader.ReadByte();
            reader.ReadByte(); // reservado

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

            using var bodyStream = new MemoryStream(rawBody);
            using var bodyReader = new BinaryReader(bodyStream, Encoding.UTF8);
            var modelState = ModelStateBinaryConverter.ReadTransformerModelState(bodyReader);

            return (modelState, header);
        }

        /// <summary>
        /// Lee únicamente el encabezado JSON de un archivo <c>.ncbm</c> (arquitectura,
        /// metadatos, si está comprimido, etc.) sin decodificar los pesos. Útil para
        /// listar o inspeccionar modelos guardados sin pagar el costo de cargarlos.
        /// </summary>
        public static ModelBinaryHeader ReadHeaderOnly(string filePath)
        {
            using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(fileStream, Encoding.UTF8);

            ushort formatVersion = ReadAndValidateFileHeader(reader, filePath);
            reader.ReadByte(); // flags
            reader.ReadByte(); // reservado

            int headerLength = reader.ReadInt32();
            var headerBytes = reader.ReadBytes(headerLength);
            var header = JsonSerializer.Deserialize<ModelBinaryHeader>(headerBytes)
                ?? throw new InvalidDataException($"No se pudo leer el encabezado del modelo '{filePath}'.");
            header.FormatVersion = formatVersion;
            return header;
        }

        /// <summary>Indica si el archivo dado parece ser un modelo <c>.ncbm</c> válido.</summary>
        public static bool IsNcbmFile(string filePath)
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
