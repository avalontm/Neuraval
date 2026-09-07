namespace Neuraval.Evolution.MarioBridge
{
    public static class MarioControllerEncoder
    {
        public static SnesButton Decode(int controller1, int controller2)
        {
            var buttons = SnesButton.None;

            if ((controller1 & 0x01) != 0) buttons |= SnesButton.Right;
            if ((controller1 & 0x02) != 0) buttons |= SnesButton.Left;
            if ((controller1 & 0x04) != 0) buttons |= SnesButton.Down;
            if ((controller1 & 0x08) != 0) buttons |= SnesButton.Up;
            if ((controller1 & 0x10) != 0) buttons |= SnesButton.Start;
            if ((controller1 & 0x20) != 0) buttons |= SnesButton.Select;
            if ((controller1 & 0x40) != 0) buttons |= SnesButton.Y;
            if ((controller1 & 0x80) != 0) buttons |= SnesButton.B;
            if ((controller2 & 0x10) != 0) buttons |= SnesButton.R;
            if ((controller2 & 0x20) != 0) buttons |= SnesButton.L;
            if ((controller2 & 0x40) != 0) buttons |= SnesButton.X;
            if ((controller2 & 0x80) != 0) buttons |= SnesButton.A;

            return buttons;
        }
    }
}