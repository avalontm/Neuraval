using System;

namespace Neuraval.Core.Training
{
    public sealed class LabelMask
    {
        private readonly bool[] _flags;

        public LabelMask(bool[] flags)
        {
            _flags = flags ?? throw new ArgumentNullException(nameof(flags));
        }

        public int Length => _flags.Length;

        public bool this[int index] => _flags[index];

        public int LabeledCount
        {
            get
            {
                int count = 0;

                for (int i = 0; i < _flags.Length; i++)
                {
                    if (_flags[i])
                        count++;
                }

                return count;
            }
        }

        public bool[] ToArray()
        {
            return (bool[])_flags.Clone();
        }

        public static LabelMask AllLabeled(int length)
        {
            return FromPredicate(length, _ => true);
        }

        public static LabelMask AllUnlabeled(int length)
        {
            return FromPredicate(length, _ => false);
        }

        public static LabelMask FromPredicate(int length, Func<int, bool> predicate)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length), "length no puede ser negativo");

            if (predicate == null)
                throw new ArgumentNullException(nameof(predicate));

            var flags = new bool[length];

            for (int i = 0; i < length; i++)
            {
                flags[i] = predicate(i);
            }

            return new LabelMask(flags);
        }
    }
}
