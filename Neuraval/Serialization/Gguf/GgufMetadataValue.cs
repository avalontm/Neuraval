using System;
using System.Collections.Generic;

namespace Neuraval.Core.Serialization.Gguf
{
    public sealed class GgufMetadataValue
    {
        public GgufValueType Type { get; }

        public object? Raw { get; }

        public GgufMetadataValue(GgufValueType type, object? raw)
        {
            Type = type;
            Raw = raw;
        }

        public bool TryGetUInt64(out ulong value)
        {
            switch (Raw)
            {
                case byte v: value = v; return true;
                case sbyte v: value = v >= 0 ? (ulong)v : 0; return v >= 0;
                case ushort v: value = v; return true;
                case short v: value = v >= 0 ? (ulong)v : 0; return v >= 0;
                case uint v: value = v; return true;
                case int v: value = v >= 0 ? (ulong)v : 0; return v >= 0;
                case ulong v: value = v; return true;
                case long v: value = v >= 0 ? (ulong)v : 0; return v >= 0;
                default: value = 0; return false;
            }
        }

        public bool TryGetBoolean(out bool value)
        {
            if (Raw is bool boolean)
            {
                value = boolean;
                return true;
            }

            value = false;
            return false;
        }

        public bool TryGetDouble(out double value)
        {
            switch (Raw)
            {
                case float v: value = v; return true;
                case double v: value = v; return true;
                default:
                    if (TryGetUInt64(out var asUInt64))
                    {
                        value = asUInt64;
                        return true;
                    }
                    value = 0;
                    return false;
            }
        }

        public bool TryGetString(out string value)
        {
            if (Raw is string s)
            {
                value = s;
                return true;
            }

            value = string.Empty;
            return false;
        }

        public bool TryGetArray(out IReadOnlyList<GgufMetadataValue> value)
        {
            if (Raw is IReadOnlyList<GgufMetadataValue> array)
            {
                value = array;
                return true;
            }

            value = Array.Empty<GgufMetadataValue>();
            return false;
        }
    }
}
