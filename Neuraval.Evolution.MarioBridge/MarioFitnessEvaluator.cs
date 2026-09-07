using System;
using Neuraval.Evolution;

namespace Neuraval.Evolution.MarioBridge
{
    // Por que termino el episodio, mas alla de "murio si/no". No lo tenemos
    // 100% verificado a nivel de direccion de RAM (ver ClassifyDeath para el
    // porque), asi que por ahora es solo para consola/estadisticas, no
    // afecta el fitness. Si mas adelante se quiere que la evolucion evite
    // mas un tipo de muerte que otro (por ejemplo, penalizar caidas al vacio
    // mas que golpes de enemigo), ese es un ajuste a proposito sobre este
    // dato, no algo que convenga inventar aca sin que alguien lo pida.
    public enum MarioDeathCause
    {
        None,
        Enemy,
        FallOrHazard
    }

    public sealed class MarioFitnessEvaluator : IFitnessEvaluator<MarioAgent, SnesState, SnesAction>
    {
        private readonly int _maxSteps;

        // Umbral de "habia algo pegado a Mario" para clasificar una muerte
        // como "por enemigo". No es una hitbox real de SMW (esas varian por
        // sprite y no las tenemos mapeadas con confianza) -- es una
        // aproximacion practica: si el sprite mas cercano en el ultimo frame
        // vivo estaba a 20px o menos (mas o menos 1 tile y un poco), es
        // razonable asumir contacto. Puede haber falsos negativos (un
        // enemigo que empujo a Mario y ya se alejo un frame antes de que
        // muriera) y no distingue lava/pinchos/munchers de una caida real
        // al vacio (ninguno de los dos tiene sprite propio) -- ambos quedan
        // como FallOrHazard. Es una heuristica sobre datos que YA leiamos de
        // forma confiable (posicion de Mario, lista de sprites), no una
        // direccion de RAM nueva sin verificar.
        private const float EnemyContactRadius = 20f;

        // Bonus grande y plano cuando el agente termina el episodio por
        // completar el nivel (no por morir ni por reset manual). bestX ya
        // premia avanzar, pero sin esto a la evolucion le da lo mismo
        // "casi llegar" que "llegar" - este empujon hace que terminar el
        // nivel sea claramente mejor que cualquier bestX intermedio.
        private const float LevelCompleteBonus = 5000f;

        // Si Mario no mejora su bestX durante esta cantidad de pasos
        // seguidos, cortamos el episodio ahi mismo en vez de esperar el
        // techo completo de _maxSteps. La red es determinista: si la
        // entrada no cambia (Mario quieto, sin sprites cerca que perturben
        // el calculo), la salida tampoco cambia, y el genoma queda en un
        // punto fijo del que nunca sale solo. Sin este corte, cada genoma
        // "congelado" gasta el episodio entero (1200 pasos) mirando a la
        // nada; con el corte, sale del cuadro rapido y el entrenamiento
        // hace mas generaciones por hora. 300 pasos (~5s a 60fps) alcanza
        // para que un genoma que si esta progresando (aunque sea lento, o
        // parado un instante esperando el timing de un salto) no se corte
        // de mas.
        private const int StagnationStepLimit = 300;

        public MarioFitnessEvaluator(int maxSteps)
        {
            _maxSteps = maxSteps;
        }

        public float Evaluate(MarioAgent agent, IEnvironment<SnesState, SnesAction> environment)
        {
            var state = environment.Reset();
            var bestX = state.MarioX;
            var stepsSinceProgress = 0;
            var deathCause = MarioDeathCause.None;

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

                // Solo clasificamos en la transicion "vivo -> muerto" (la
                // primera vez que lo vemos), usando el ultimo estado vivo:
                // una vez muerto, la posicion de Mario y la lista de sprites
                // ya no reflejan el momento del golpe, sino la animacion de
                // muerte en curso.
                if (state.IsDead && deathCause == MarioDeathCause.None)
                {
                    deathCause = ClassifyDeath(previousState);
                }

                // El episodio se corta si Mario muere, si completa el nivel
                // (llega a la meta), si se pidio un reset manual desde
                // BizHawk, o si se quedo sin progresar demasiado tiempo
                // (genoma "congelado" en un punto fijo). _maxSteps sigue
                // siendo un techo de seguridad para no quedarse atado a un
                // agente para siempre si nada de eso pasa.
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

            if (deathCause != MarioDeathCause.None)
            {
                var causeLabel = deathCause == MarioDeathCause.Enemy ? "enemigo" : "caida/peligro (pozo, lava, pinchos...)";
                Console.WriteLine($"    Murio por: {causeLabel} (bestX={bestX})");
            }

            return fitness;
        }

        private static MarioDeathCause ClassifyDeath(SnesState lastLivingState)
        {
            var nearestDistanceSquared = float.MaxValue;

            foreach (var sprite in lastLivingState.Sprites)
            {
                var dx = sprite.X - lastLivingState.MarioX;
                var dy = sprite.Y - lastLivingState.MarioY;
                var distanceSquared = dx * dx + dy * dy;

                if (distanceSquared < nearestDistanceSquared)
                {
                    nearestDistanceSquared = distanceSquared;
                }
            }

            return nearestDistanceSquared <= EnemyContactRadius * EnemyContactRadius
                ? MarioDeathCause.Enemy
                : MarioDeathCause.FallOrHazard;
        }
    }
}
