using Neuraval.Evolution;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioFitnessEvaluator : IFitnessEvaluator<MarioAgent, SnesState, SnesAction>
    {
        private readonly int _maxSteps;

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

            for (var step = 0; step < _maxSteps; step++)
            {
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

            return fitness;
        }
    }
}
