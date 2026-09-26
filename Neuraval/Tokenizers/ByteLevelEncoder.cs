namespace Neuraval.Core.Tokenizers
{
    public static class ByteLevelEncoder
    {
        private static readonly Dictionary<byte, char> ByteToChar = BuildByteToChar();
        private static readonly Dictionary<char, byte> CharToByte = BuildCharToByte();

        public static IReadOnlyCollection<char> Alphabet => ByteToChar.Values;

        private static Dictionary<byte, char> BuildByteToChar()
        {
            var printable = new HashSet<int>();

            for (int b = '!'; b <= '~'; b++)
                printable.Add(b);

            for (int b = 0xA1; b <= 0xAC; b++)
                printable.Add(b);

            for (int b = 0xAE; b <= 0xFF; b++)
                printable.Add(b);

            var map = new Dictionary<byte, char>();

            foreach (var b in printable)
                map[(byte)b] = (char)b;

            int next = 0;

            for (int b = 0; b < 256; b++)
            {
                if (!map.ContainsKey((byte)b))
                {
                    map[(byte)b] = (char)(256 + next);
                    next++;
                }
            }

            return map;
        }

        private static Dictionary<char, byte> BuildCharToByte()
        {
            var reversed = new Dictionary<char, byte>();

            foreach (var pair in ByteToChar)
                reversed[pair.Value] = pair.Key;

            return reversed;
        }

        public static string Encode(string text)
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(text);
            var builder = new System.Text.StringBuilder(bytes.Length);

            foreach (var b in bytes)
                builder.Append(ByteToChar[b]);

            return builder.ToString();
        }

        public static string Decode(string encoded)
        {
            var bytes = new byte[encoded.Length];

            for (int i = 0; i < encoded.Length; i++)
            {
                if (!CharToByte.TryGetValue(encoded[i], out var b))
                    throw new ArgumentException($"El carácter '{encoded[i]}' no pertenece al alfabeto byte-level");

                bytes[i] = b;
            }

            return System.Text.Encoding.UTF8.GetString(bytes);
        }
    }
}
