using System;

namespace Neuraval.Evolution
{
    public static class RandomExtensions
    {
        public static float NextGaussian(this Random random, float mean, float standardDeviation)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double standardNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + standardDeviation * (float)standardNormal;
        }
    }
}
