using System;

namespace Neuraval.Evolution.MarioBridge
{
    [Flags]
    public enum SnesButton
    {
        None = 0,
        A = 1 << 0,
        B = 1 << 1,
        X = 1 << 2,
        Y = 1 << 3,
        Up = 1 << 4,
        Down = 1 << 5,
        Left = 1 << 6,
        Right = 1 << 7,
        L = 1 << 8,
        R = 1 << 9,
        Select = 1 << 10,
        Start = 1 << 11
    }
}
