namespace Neuraval.Evolution.Serialization
{
    [Flags]
    public enum NavmBinaryFlags : byte
    {
        None = 0,
        GZipCompressed = 1 << 0
    }

    public static class NavmBinaryFormat
    {
        public const string FileExtension = ".navm";
        public static readonly byte[] MagicBytes = { (byte)'N', (byte)'A', (byte)'V', (byte)'M' };
        public const ushort CurrentFormatVersion = 1;
        public const int ChecksumLength = 32;
    }
}