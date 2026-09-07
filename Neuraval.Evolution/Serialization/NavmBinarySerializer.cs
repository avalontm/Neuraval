using System.IO;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;

namespace Neuraval.Evolution.Serialization
{
    public sealed class NavmBinaryContent
    {
        public string HeaderJson { get; init; }
        public byte[] Body { get; init; }

        public NavmBinaryContent(string headerJson, byte[] body)
        {
            HeaderJson = headerJson;
            Body = body;
        }
    }

    public static class NavmBinarySerializer
    {
        public static void Save(string filePath, string headerJson, byte[] body, bool compress = true)
        {
            if (filePath == null) throw new ArgumentNullException(nameof(filePath));
            if (headerJson == null) throw new ArgumentNullException(nameof(headerJson));
            if (body == null) throw new ArgumentNullException(nameof(body));

            var directory = Path.GetDirectoryName(Path.GetFullPath(filePath));
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var flags = NavmBinaryFlags.None;
            byte[] bodyOnDisk;
            if (compress)
            {
                using var compressedStream = new MemoryStream();
                using (var gzip = new GZipStream(compressedStream, CompressionLevel.Optimal, leaveOpen: true))
                {
                    gzip.Write(body, 0, body.Length);
                }

                bodyOnDisk = compressedStream.ToArray();
                flags |= NavmBinaryFlags.GZipCompressed;
            }
            else
            {
                bodyOnDisk = body;
            }

            var checksum = SHA256.HashData(bodyOnDisk);
            var headerBytes = Encoding.UTF8.GetBytes(headerJson);

            var tempPath = filePath + ".tmp";
            using (var stream = new FileStream(tempPath, FileMode.Create, FileAccess.Write))
            using (var writer = new BinaryWriter(stream, Encoding.UTF8))
            {
                writer.Write(NavmBinaryFormat.MagicBytes);
                writer.Write(NavmBinaryFormat.CurrentFormatVersion);
                writer.Write((byte)flags);
                writer.Write((byte)0);
                writer.Write(headerBytes.Length);
                writer.Write(headerBytes);
                writer.Write(bodyOnDisk.Length);
                writer.Write(bodyOnDisk);
                writer.Write(checksum);
            }

            File.Move(tempPath, filePath, overwrite: true);
        }

        public static NavmBinaryContent? Load(string filePath)
        {
            try
            {
                using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
                return Load(stream);
            }
            catch (IOException)
            {
                return null;
            }
        }

        public static NavmBinaryContent? Load(Stream stream)
        {
            try
            {
                using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

                var magic = reader.ReadBytes(NavmBinaryFormat.MagicBytes.Length);
                if (magic.Length != NavmBinaryFormat.MagicBytes.Length || !magic.AsSpan().SequenceEqual(NavmBinaryFormat.MagicBytes))
                {
                    return null;
                }

                var version = reader.ReadUInt16();
                if (version > NavmBinaryFormat.CurrentFormatVersion)
                {
                    return null;
                }

                var flags = (NavmBinaryFlags)reader.ReadByte();
                reader.ReadByte();

                var headerLength = reader.ReadInt32();
                var headerBytes = reader.ReadBytes(headerLength);
                if (headerBytes.Length != headerLength)
                {
                    return null;
                }

                var bodyLength = reader.ReadInt32();
                var bodyOnDisk = reader.ReadBytes(bodyLength);
                if (bodyOnDisk.Length != bodyLength)
                {
                    return null;
                }

                var storedChecksum = reader.ReadBytes(NavmBinaryFormat.ChecksumLength);
                if (storedChecksum.Length != NavmBinaryFormat.ChecksumLength)
                {
                    return null;
                }

                var actualChecksum = SHA256.HashData(bodyOnDisk);
                if (!storedChecksum.AsSpan().SequenceEqual(actualChecksum))
                {
                    return null;
                }

                var body = flags.HasFlag(NavmBinaryFlags.GZipCompressed) ? Decompress(bodyOnDisk) : bodyOnDisk;
                return new NavmBinaryContent(Encoding.UTF8.GetString(headerBytes), body);
            }
            catch (Exception ex) when (ex is IOException || ex is InvalidDataException || ex is EndOfStreamException)
            {
                return null;
            }
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