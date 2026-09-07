namespace Neuraval.Evolution.MarioBridge
{
    public enum MarioDeathCause
    {
        None,
        Enemy,
        FallOrHazard
    }

    public sealed class MarioFitnessEvaluator : IFitnessEvaluator<MarioAgent, SnesState, SnesAction>
    {
        private const float EnemyContactRadius = 20f;
        private const float LevelCompleteBonus = 5000f;
        private const float PowerupGainWorth = 80f;
        private const float CoinWorth = 25f;
        private const float DamagePenalty = 20f;
        private const float DeathPenalty = 250f;
        private const int StagnationStepLimit = 300;

        // Monedas Yoshi ($7E:1420): valen mucho mas que una moneda normal
        // porque suelen requerir desviarse del camino/explorar, y coleccionar
        // las 5 de un nivel da un bonus extra grande para incentivar barrer
        // el nivel completo en vez de solo correr a la meta.
        private const float YoshiCoinWorth = 150f;
        private const int YoshiCoinsPerLevel = 5;
        private const float AllYoshiCoinsBonus = 1000f;

        // Punto medio / checkpoint ($7E:13CE): recompensa por activarlo, para
        // que la IA aprenda a pisar la barra de mitad de nivel (guarda el
        // progreso de respawn) en vez de ignorarla.
        private const float MidwayPointBonus = 300f;

        private readonly int _maxSteps;

        public MarioFitnessEvaluator(int maxSteps)
        {
            _maxSteps = maxSteps;
        }

        public float Evaluate(MarioAgent agent, IEnvironment<SnesState, SnesAction> environment)
        {
            var state = environment.Reset();
            var bestX = state.MarioX;
            var stepsSinceProgress = 0;
            var deathInfo = NoDeath;

            var previousPowerup = state.PowerupLevel;
            var previousCoins = state.Coins;
            var previousHurt = state.HurtTimer;
            var previousYoshiCoins = state.YoshiCoinsCollected;
            var previousMidway = state.MidwayPointReached;
            var powerupGains = 0;
            var coinGains = 0;
            var damageHits = 0;
            var yoshiCoinGains = 0;
            var maxYoshiCoinsSeen = state.YoshiCoinsCollected;
            var reachedMidway = state.MidwayPointReached;

            for (var step = 0; step < _maxSteps; step++)
            {
                var previousState = state;
                var action = agent.Decide(state);
                var result = environment.Step(action);
                state = result.State;

                if (state.MarioX > bestX)
                {
                    bestX = state.MarioX;
                    stepsSinceProgress = 0;
                }
                else
                {
                    stepsSinceProgress++;
                }

                var powerupDelta = state.PowerupLevel - previousPowerup;
                if (powerupDelta > 0)
                {
                    powerupGains += powerupDelta;
                }

                var coinDelta = state.Coins - previousCoins;
                if (coinDelta > 0)
                {
                    coinGains += coinDelta;
                }

                if (state.HurtTimer > 0 && previousHurt == 0)
                {
                    damageHits++;
                }

                var yoshiCoinDelta = state.YoshiCoinsCollected - previousYoshiCoins;
                if (yoshiCoinDelta > 0)
                {
                    yoshiCoinGains += yoshiCoinDelta;
                }

                if (state.YoshiCoinsCollected > maxYoshiCoinsSeen)
                {
                    maxYoshiCoinsSeen = state.YoshiCoinsCollected;
                }

                if (state.MidwayPointReached && !previousMidway)
                {
                    reachedMidway = true;
                }

                previousPowerup = state.PowerupLevel;
                previousCoins = state.Coins;
                previousHurt = state.HurtTimer;
                previousYoshiCoins = state.YoshiCoinsCollected;
                previousMidway = state.MidwayPointReached;

                if (state.IsDead && deathInfo.Cause == MarioDeathCause.None)
                {
                    deathInfo = ClassifyDeath(previousState);
                }

                if (result.Done || stepsSinceProgress >= StagnationStepLimit)
                {
                    break;
                }
            }

            float fitness = bestX;

            if (state.IsLevelComplete)
            {
                fitness += LevelCompleteBonus;
            }

            fitness += powerupGains * PowerupGainWorth;
            fitness += coinGains * CoinWorth;
            fitness += yoshiCoinGains * YoshiCoinWorth;
            fitness -= damageHits * DamagePenalty;

            if (maxYoshiCoinsSeen >= YoshiCoinsPerLevel)
            {
                fitness += AllYoshiCoinsBonus;
            }

            if (reachedMidway)
            {
                fitness += MidwayPointBonus;
            }

            if (deathInfo.Cause != MarioDeathCause.None)
            {
                fitness -= DeathPenalty;
                var causeLabel = deathInfo.Cause == MarioDeathCause.Enemy
                    ? $"enemigo {MarioSpriteNames.Name(deathInfo.SpriteType)} (${deathInfo.SpriteType:X2}) cerca de X={deathInfo.EnemyX}, Y={deathInfo.EnemyY}"
                    : "caida/peligro (pozo, lava, pinchos...)";
                Console.WriteLine($"    Murio por: {causeLabel} (bestX={bestX})");
            }

            return fitness;
        }

        private readonly record struct DeathInfo(MarioDeathCause Cause, int SpriteType, int EnemyX, int EnemyY);

        private static readonly DeathInfo NoDeath = new(MarioDeathCause.None, 0, 0, 0);

        private static DeathInfo ClassifyDeath(SnesState lastLivingState)
        {
            var nearestDistanceSquared = float.MaxValue;
            var nearestType = 0;
            var nearestX = 0;
            var nearestY = 0;

            foreach (var sprite in lastLivingState.Sprites)
            {
                var dx = sprite.X - lastLivingState.MarioX;
                var dy = sprite.Y - lastLivingState.MarioY;
                var distanceSquared = dx * dx + dy * dy;

                if (distanceSquared < nearestDistanceSquared)
                {
                    nearestDistanceSquared = distanceSquared;
                    nearestType = sprite.Type;
                    nearestX = sprite.X;
                    nearestY = sprite.Y;
                }
            }

            return nearestDistanceSquared <= EnemyContactRadius * EnemyContactRadius
                ? new DeathInfo(MarioDeathCause.Enemy, nearestType, nearestX, nearestY)
                : new DeathInfo(MarioDeathCause.FallOrHazard, nearestType, nearestX, nearestY);
        }
    }
}