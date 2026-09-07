namespace Neuraval.Evolution.MarioBridge
{
    public static class MarioAgentOutput
    {
        public const int Count = 7;
        public const int LeftIndex = 0;
        public const int RightIndex = 1;
        public const int AIndex = 2;
        public const int BIndex = 3;
        public const int YIndex = 4;
        public const int DownIndex = 5;
        public const int UpIndex = 6;

        public static SnesAction ToAction(float[] output)
        {
            var buttons = SnesButton.None;

            if (output[LeftIndex] > 0f || output[RightIndex] > 0f)
            {
                buttons |= output[LeftIndex] >= output[RightIndex] ? SnesButton.Left : SnesButton.Right;
            }

            if (output[AIndex] > 0f) buttons |= SnesButton.A;
            if (output[BIndex] > 0f) buttons |= SnesButton.B;
            if (output[YIndex] > 0f) buttons |= SnesButton.Y;
            if (output[DownIndex] > 0f) buttons |= SnesButton.Down;
            if (output[UpIndex] > 0f) buttons |= SnesButton.Up;

            return new SnesAction(buttons);
        }

        public static void ToAgentTargets(SnesButton mask, Span<float> targets)
        {
            targets[LeftIndex] = mask.HasFlag(SnesButton.Left) ? 1f : 0f;
            targets[RightIndex] = mask.HasFlag(SnesButton.Right) ? 1f : 0f;
            targets[AIndex] = mask.HasFlag(SnesButton.A) ? 1f : 0f;
            targets[BIndex] = mask.HasFlag(SnesButton.B) ? 1f : 0f;
            targets[YIndex] = mask.HasFlag(SnesButton.Y) ? 1f : 0f;
            targets[DownIndex] = mask.HasFlag(SnesButton.Down) ? 1f : 0f;
            targets[UpIndex] = mask.HasFlag(SnesButton.Up) ? 1f : 0f;
        }
    }
}