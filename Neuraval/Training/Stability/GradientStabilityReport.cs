namespace Neuraval.Core.Training.Stability
{
    public sealed class GradientStabilityReport
    {
        private GradientStabilityReport(
            float normBeforeClip,
            float normAfterClip,
            bool wasClipped,
            bool hasNaN,
            bool hasInfinity)
        {
            NormBeforeClip = normBeforeClip;
            NormAfterClip = normAfterClip;
            WasClipped = wasClipped;
            HasNaN = hasNaN;
            HasInfinity = hasInfinity;
        }

        public float NormBeforeClip { get; }

        public float NormAfterClip { get; }

        public bool WasClipped { get; }

        public bool HasNaN { get; }

        public bool HasInfinity { get; }

        public bool IsStable => !HasNaN && !HasInfinity;

        public static GradientStabilityReport Create(
            float normBeforeClip,
            float normAfterClip,
            bool wasClipped,
            bool hasNaN,
            bool hasInfinity)
        {
            return new GradientStabilityReport(normBeforeClip, normAfterClip, wasClipped, hasNaN, hasInfinity);
        }
    }
}
