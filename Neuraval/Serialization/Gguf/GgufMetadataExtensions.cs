using System.Collections.Generic;

namespace Neuraval.Core.Serialization.Gguf
{
    public static class GgufMetadataExtensions
    {
        public static bool TryGetUInt64(this IReadOnlyDictionary<string, GgufMetadataValue> metadata, string key, out ulong value)
        {
            if (metadata.TryGetValue(key, out var entry))
                return entry.TryGetUInt64(out value);

            value = 0;
            return false;
        }

        public static bool TryGetInt32(this IReadOnlyDictionary<string, GgufMetadataValue> metadata, string key, out int value)
        {
            if (metadata.TryGetUInt64(key, out var raw))
            {
                value = (int)raw;
                return true;
            }

            value = 0;
            return false;
        }

        public static bool TryGetFloat(this IReadOnlyDictionary<string, GgufMetadataValue> metadata, string key, out float value)
        {
            if (metadata.TryGetValue(key, out var entry) && entry.TryGetDouble(out var raw))
            {
                value = (float)raw;
                return true;
            }

            value = 0f;
            return false;
        }

        public static bool TryGetString(this IReadOnlyDictionary<string, GgufMetadataValue> metadata, string key, out string value)
        {
            if (metadata.TryGetValue(key, out var entry))
                return entry.TryGetString(out value);

            value = string.Empty;
            return false;
        }

        public static bool TryGetArrayLength(this IReadOnlyDictionary<string, GgufMetadataValue> metadata, string key, out int length)
        {
            if (metadata.TryGetValue(key, out var entry) && entry.TryGetArray(out var array))
            {
                length = array.Count;
                return true;
            }

            length = 0;
            return false;
        }
    }
}
