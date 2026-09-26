using System;

namespace Neuraval.Core.Serialization.SafeTensors
{
    public enum SafeTensorsDType
    {
        F32,
        F16,
        BF16
    }

    public static class SafeTensorsDTypeExtensions
    {
        public static string ToTag(this SafeTensorsDType dtype)
        {
            return dtype switch
            {
                SafeTensorsDType.F32 => "F32",
                SafeTensorsDType.F16 => "F16",
                SafeTensorsDType.BF16 => "BF16",
                _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, null)
            };
        }

        public static SafeTensorsDType FromTag(string tag)
        {
            if (tag == null)
                throw new ArgumentNullException(nameof(tag));

            return tag switch
            {
                "F32" => SafeTensorsDType.F32,
                "F16" => SafeTensorsDType.F16,
                "BF16" => SafeTensorsDType.BF16,
                _ => throw new NotSupportedException($"dtype '{tag}' no soportado")
            };
        }

        public static int ByteSize(this SafeTensorsDType dtype)
        {
            return dtype switch
            {
                SafeTensorsDType.F32 => 4,
                SafeTensorsDType.F16 => 2,
                SafeTensorsDType.BF16 => 2,
                _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, null)
            };
        }
    }
}
