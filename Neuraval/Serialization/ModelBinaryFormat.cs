using System;

namespace Neuraval.Core.Serialization
{
    public static class ModelBinaryFormat
    {
        public const string FileExtension = ".navm";

        public static readonly byte[] MagicBytes = { (byte)'N', (byte)'A', (byte)'V', (byte)'M' };

        public const ushort CurrentFormatVersion = 2;

        public const int ChecksumLength = 32;

        [Flags]
        public enum ModelFlags : byte
        {
            None = 0,
            GZipCompressed = 1 << 0,

            Int8QuantizedWeights = 1 << 1
        }
    }
}
