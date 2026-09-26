using System;

namespace Neuraval.Core.Training.Precision
{
    public readonly struct BFloat16 : IEquatable<BFloat16>
    {
        private readonly ushort _bits;

        private BFloat16(ushort bits)
        {
            _bits = bits;
        }

        public ushort RawBits => _bits;

        public static BFloat16 FromSingle(float value)
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);

            if (float.IsNaN(value))
            {
                return new BFloat16((ushort)((bits >> 16) | 0x0040));
            }

            uint roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
            uint rounded = bits + roundingBias;

            return new BFloat16((ushort)(rounded >> 16));
        }

        public float ToSingle()
        {
            uint bits = (uint)_bits << 16;
            return BitConverter.UInt32BitsToSingle(bits);
        }

        public static implicit operator BFloat16(float value) => FromSingle(value);

        public static implicit operator float(BFloat16 value) => value.ToSingle();

        public bool Equals(BFloat16 other) => _bits == other._bits;

        public override bool Equals(object? obj) => obj is BFloat16 other && Equals(other);

        public override int GetHashCode() => _bits.GetHashCode();
    }
}
