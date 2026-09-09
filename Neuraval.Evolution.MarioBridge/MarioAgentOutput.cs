namespace Neuraval.Evolution.MarioBridge
{
    public static class MarioAgentOutput
    {
        public const int Count = 9;
        public const int LeftIndex = 0;
        public const int RightIndex = 1;
        public const int AIndex = 2;
        public const int BIndex = 3;
        public const int YIndex = 4;
        public const int DownIndex = 5;
        public const int UpIndex = 6;
        public const int XIndex = 7;
        public const int RIndex = 8;
        public const float ButtonThreshold = 0.5f;

        public static SnesAction ToAction(float[] output)
        {
            var buttons = SnesButton.None;

            if (output[LeftIndex] >= ButtonThreshold || output[RightIndex] >= ButtonThreshold)
            {
                buttons |= output[LeftIndex] >= output[RightIndex] ? SnesButton.Left : SnesButton.Right;
            }

            if (output[AIndex] >= ButtonThreshold) buttons |= SnesButton.A;
            if (output[BIndex] >= ButtonThreshold) buttons |= SnesButton.B;
            if (output[YIndex] >= ButtonThreshold) buttons |= SnesButton.Y;
            if (output[DownIndex] >= ButtonThreshold) buttons |= SnesButton.Down;
            if (output[UpIndex] >= ButtonThreshold) buttons |= SnesButton.Up;
            if (output[XIndex] >= ButtonThreshold) buttons |= SnesButton.X;
            if (output[RIndex] >= ButtonThreshold) buttons |= SnesButton.R;

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
            targets[XIndex] = mask.HasFlag(SnesButton.X) ? 1f : 0f;
            targets[RIndex] = mask.HasFlag(SnesButton.R) ? 1f : 0f;
        }
    }
}