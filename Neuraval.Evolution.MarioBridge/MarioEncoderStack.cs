using System;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioEncoderStack
    {
        private readonly float[][] _ring;
        private int _count;

        public MarioEncoderStack()
        {
            _ring = new float[MarioStateEncoder.FrameHistory][];
            for (var i = 0; i < _ring.Length; i++)
            {
                _ring[i] = new float[MarioStateEncoder.InputCount];
            }
        }

        public void Reset()
        {
            _count = 0;
        }

        public float[] Encode(SnesState state, float[]? destination = null)
        {
            var slot = _ring[_count % _ring.Length];
            MarioStateEncoder.EncodeFrameVector(state, slot);
            _count++;

            var result = destination ?? new float[MarioStateEncoder.StackedInputCount];
            Array.Clear(result, 0, result.Length);

            var available = Math.Min(_ring.Length, _count);
            for (var k = 0; k < available; k++)
            {
                var time = _count - available + k;
                var offset = (MarioStateEncoder.FrameHistory - available + k) * MarioStateEncoder.InputCount;
                Array.Copy(_ring[time % _ring.Length], 0, result, offset, MarioStateEncoder.InputCount);
            }

            return result;
        }
    }
}