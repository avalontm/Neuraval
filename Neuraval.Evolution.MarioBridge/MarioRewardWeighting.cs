using System;

namespace Neuraval.Evolution.MarioBridge
{
    public static class MarioRewardWeighting
    {
        public static float Compute(float reward, float minReward, float maxReward, float strength)
        {
            if (maxReward <= minReward || strength <= 0f)
            {
                return 1f;
            }

            var normalized = (reward - minReward) / (maxReward - minReward);
            return 1f + strength * Math.Clamp(normalized, 0f, 1f);
        }
    }
}