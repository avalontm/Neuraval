using System;

namespace Neuraval.Core.Serialization
{
    public static class LoraBinaryFormat
    {
        public const string FileExtension = ".navlora";

        public static readonly byte[] MagicBytes = { (byte)'N', (byte)'A', (byte)'V', (byte)'L' };

        public const ushort CurrentFormatVersion = 1;

        public const int ChecksumLength = 32;

        [Flags]
        public enum LoraFlags : byte
        {
            None = 0,
            GZipCompressed = 1 << 0
        }
    }
}
