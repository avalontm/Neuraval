using System;
using System.Threading;

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
        private const int FallScreenBottomMargin = 224;

        private const float LevelCompleteBonus = 5000f;
        private const float DeathPenalty = 400f;
        private const float PowerupGainWorth = 40f;
        private const float DamagePenalty = 40f;
        private const float CoinWorth = 5f;
        private const float YoshiCoinWorth = 30f;
        private const int YoshiCoinsPerLevel = 5;
        private const float AllYoshiCoinsBonus = 150f;
        private const float MidwayPointBonus = 50f;

        private const int NoMovementStepLimit = 300;

        private readonly int _maxSteps;
        private int _completions;
        private int _deathsByEnemy;
        private int _deathsByFall;
        private int _stepsCapTerminations;
        private int _bestXThisGeneration;

        public int Completions => _completions;
        public int DeathsByEnemy => _deathsByEnemy;
        public int DeathsByFall => _deathsByFall;
        public int StepsCapTerminations => _stepsCapTerminations;
        public int BestXThisGeneration => _bestXThisGeneration;

        public void ResetCounters()
        {
            Interlocked.Exchange(ref _completions, 0);
            Interlocked.Exchange(ref _deathsByEnemy, 0);
            Interlocked.Exchange(ref _deathsByFall, 0);
            Interlocked.Exchange(ref _stepsCapTerminations, 0);
            Interlocked.Exchange(ref _bestXThisGeneration, 0);
        }

        public MarioFitnessEvaluator(int maxSteps)
        {
            _maxSteps = maxSteps;
        }

        public float Evaluate(MarioAgent agent, IEnvironment<SnesState, SnesAction> environment)
        {
            var state = environment.Reset();
            agent.ResetHistory();
            var bestX = state.MarioX;
            var stepsWithoutMovement = 0;
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

            var step = 0;
            for (; step < _maxSteps; step++)
            {
                var previousState = state;
                var action = agent.Decide(state);
                var result = environment.Step(action);
                state = result.State;

                if (state.MarioX > bestX)
                {
                    bestX = state.MarioX;
                }

                if (state.MarioX != previousState.MarioX || state.MarioY != previousState.MarioY)
                {
                    stepsWithoutMovement = 0;
                }
                else
                {
                    stepsWithoutMovement++;
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

                if (result.Done || stepsWithoutMovement >= NoMovementStepLimit)
                {
                    break;
                }
            }

            UpdateMax(ref _bestXThisGeneration, bestX);

            if (state.IsLevelComplete)
            {
                Interlocked.Increment(ref _completions);
            }
            else if (deathInfo.Cause == MarioDeathCause.Enemy)
            {
                Interlocked.Increment(ref _deathsByEnemy);
            }
            else if (deathInfo.Cause == MarioDeathCause.FallOrHazard)
            {
                Interlocked.Increment(ref _deathsByFall);
            }
            else if (step >= _maxSteps || state.ManualResetRequested)
            {
                Interlocked.Increment(ref _stepsCapTerminations);
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

        public static string DescribeDeath(SnesState lastLivingState)
        {
            var info = ClassifyDeath(lastLivingState);
            if (info.Cause == MarioDeathCause.None)
            {
                return string.Empty;
            }

            return info.Cause == MarioDeathCause.Enemy
                ? $"enemigo {MarioSpriteNames.Name(info.SpriteType)} (${info.SpriteType:X2}) cerca de X={info.EnemyX}, Y={info.EnemyY}"
                : "caida/peligro (pozo, lava, pinchos...)";
        }

        public static MarioDeathCause ClassifyDeathCause(SnesState state)
        {
            return ClassifyDeath(state).Cause;
        }

        private static DeathInfo ClassifyDeath(SnesState lastLivingState)
        {
            var (nearestType, nearestX, nearestY, nearestDistanceSquared) = NearestSpriteWithDistance(lastLivingState);

            var fellBelowScreen = lastLivingState.MarioY - lastLivingState.CameraY >= FallScreenBottomMargin;
            if (fellBelowScreen)
            {
                return new DeathInfo(MarioDeathCause.FallOrHazard, nearestType, nearestX, nearestY);
            }

            var enemyInContactRange = nearestDistanceSquared <= EnemyContactRadius * EnemyContactRadius;

            return enemyInContactRange
                ? new DeathInfo(MarioDeathCause.Enemy, nearestType, nearestX, nearestY)
                : new DeathInfo(MarioDeathCause.FallOrHazard, nearestType, nearestX, nearestY);
        }

        private static (int Type, int X, int Y, float DistanceSquared) NearestSpriteWithDistance(SnesState state)
        {
            var nearestDistanceSquared = float.MaxValue;
            var nearestType = 0;
            var nearestX = 0;
            var nearestY = 0;

            foreach (var sprite in state.Sprites)
            {
                var dx = sprite.X - state.MarioX;
                var dy = sprite.Y - state.MarioY;
                var distanceSquared = dx * dx + dy * dy;

                if (distanceSquared < nearestDistanceSquared)
                {
                    nearestDistanceSquared = distanceSquared;
                    nearestType = sprite.Type;
                    nearestX = sprite.X;
                    nearestY = sprite.Y;
                }
            }

            return (nearestType, nearestX, nearestY, nearestDistanceSquared);
        }

        private static void UpdateMax(ref int target, int value)
        {
            int initial, computed;
            do
            {
                initial = target;
                computed = Math.Max(initial, value);
            } while (Interlocked.CompareExchange(ref target, computed, initial) != initial);
        }
    }
}
