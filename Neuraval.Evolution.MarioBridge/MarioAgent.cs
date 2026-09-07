using System;
using System.Collections.Generic;
using System.Linq;
using Neuraval.Evolution;
using Neuraval.Evolution.Neat;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioAgent : IAgent<SnesState, SnesAction>, INeatAgent<MarioAgent>
    {
        private const int GridCellCount = SnesState.GridSize * SnesState.GridSize;

        // Cuantos sprites cercanos le pasamos a la red (antes: 1, sin tipo).
        // Cada sprite aporta 3 señales: dx, dy y su tipo (que enemigo/item es).
        private const int TrackedSpriteCount = 3;
        private const int SignalsPerSprite = 3;
        private const int VelocitySignalCount = 2;

        // Señal directa de "estoy parado en el piso o en el aire". Antes la
        // red tenia que inferir esto indirectamente de MarioVelocityY (que
        // ademas comparte magnitud con la velocidad de caida, la de subida
        // del salto, etc.) - eso le pedia mucho trabajo a la evolucion para
        // aprender a distinguir salto corto (soltar B apenas despega) de
        // salto largo (sostener B mientras sube). Con esta entrada explicita
        // (1 = piso, 0 = aire) la red tiene una señal limpia para condicionar
        // esa decision.
        private const int GroundedSignalCount = 1;

        // Forma actual de Mario (chico/grande/capa/fuego), codificada
        // one-hot (4 señales, una prende y el resto en 0) en vez de un solo
        // numero normalizado. Con un escalar (ej: powerup/3f) la red
        // asumiria una relacion de orden entre las formas -- pero capa (2) y
        // fuego (3) no son "mas" o "menos" que el otro, son estrategias
        // distintas (planear vs atacar a distancia) que conviene que la red
        // pueda distinguir sin forzar una escala. Esto tambien importa para
        // el boton Y: con Mario de fuego, Y tira una bola de fuego en vez de
        // correr/sostener un item -- la red necesita saber en que forma esta
        // para aprender cuando conviene apretarlo.
        private const int PowerupSignalCount = 4;
        private const int MaxKnownPowerupLevel = PowerupSignalCount - 1;

        private const int ExtraSignalCount = VelocitySignalCount + GroundedSignalCount + PowerupSignalCount + TrackedSpriteCount * SignalsPerSprite;

        public const int InputCount = GridCellCount + ExtraSignalCount;

        // Nota: el SnesState ahora trae LevelIndex (que nivel/savestate esta
        // corriendo), pero deliberadamente NO lo metemos como entrada de la
        // red. Dos motivos: (1) la grilla de tiles y los sprites cercanos ya
        // le dan a la red toda la informacion que necesita para reaccionar
        // al nivel en el que esta -- igual que un jugador humano no necesita
        // que le digan "estas en el nivel 3", lee el entorno; (2) el numero
        // de niveles configurados vive en mario_bridge.lua (SAVESTATE_FILES)
        // y en Program.cs (MarioLevels) por separado -- convertirlo en
        // entrada de red exigiria mantener esos dos archivos perfectamente
        // sincronizados en cantidad, o la topologia de la red quedaria mal
        // formada sin que nada lo avise. Si en el futuro hace falta que la
        // red distinga niveles explicitamente (por ejemplo, si dos niveles
        // se ven demasiado parecidos en la grilla local), se puede agregar
        // como input igual que se hizo con IsGrounded -- pero por ahora no
        // hay evidencia de que haga falta.

        // 5ta salida: Y. En SMW default, Y es "correr" (sostenido, aumenta
        // velocidad) y tambien "agarrar/sostener/soltar" un item (shell, etc)
        // cuando Mario esta al lado de uno - es el mismo boton fisico, el
        // contexto (hay o no un item pegado a Mario) es lo que determina el
        // efecto en el juego. No separamos "correr" de "sostener item" como
        // dos salidas distintas por la misma razon que no clasificamos tipos
        // de sprite a mano: dejamos que la evolucion aprenda cuando conviene
        // mantenerlo apretado.
        public const int OutputCount = 5;
        private const float VelocityScale = 16f;
        private const float SpriteOffsetScale = 100f;

        // ID de sprite mas alto conocido en SMW vanilla es 0xFF; normalizamos
        // a algo chico para que no domine el resto de las entradas.
        private const float SpriteTypeScale = 255f;

        public NeatGenome Genome { get; }

        public MarioAgent(NeatGenome genome)
        {
            Genome = genome;
        }

        public static MarioAgent CreateRandom(Random random, NeatInnovationTracker tracker)
        {
            return new MarioAgent(NeatGenome.CreateInitial(InputCount, OutputCount, random, tracker));
        }

        public MarioAgent WithGenome(NeatGenome genome)
        {
            return new MarioAgent(genome);
        }

        public SnesAction Decide(SnesState state)
        {
            var input = BuildInput(state);
            var output = Genome.Evaluate(input);

            var buttons = SnesButton.None;

            // Izquierda y Derecha se evaluaban de forma independiente antes:
            // si ambas salidas daban >0 en el mismo frame (algo comun con
            // tanh y pesos chicos/random), la red terminaba apretando las
            // dos a la vez, que en SMW se cancelan entre si -- Mario se
            // queda temblando en el lugar en vez de caminar. Ahora la red
            // elige una sola direccion: la que tenga la salida mas alta,
            // y solo si esa salida es positiva (si ninguna de las dos
            // supera 0, no se aprieta ninguna, como antes).
            if (output[0] > 0f || output[1] > 0f)
            {
                buttons |= output[0] >= output[1] ? SnesButton.Left : SnesButton.Right;
            }

            if (output[2] > 0f) buttons |= SnesButton.A; // salto girando (spin jump)
            if (output[3] > 0f) buttons |= SnesButton.B; // salto normal
            if (output[4] > 0f) buttons |= SnesButton.Y; // correr / agarrar-sostener-soltar item (segun contexto)

            return new SnesAction(buttons);
        }

        private static float[] BuildInput(SnesState state)
        {
            var input = new float[InputCount];

            for (var i = 0; i < GridCellCount; i++)
            {
                input[i] = state.Tiles[i] ? 1f : 0f;
            }

            input[GridCellCount] = state.MarioVelocityX / VelocityScale;
            input[GridCellCount + 1] = state.MarioVelocityY / VelocityScale;
            input[GridCellCount + 2] = state.IsGrounded ? 1f : 0f;

            var powerupInputBase = GridCellCount + VelocitySignalCount + GroundedSignalCount;

            // Clamp defensivo: si algun dia $19 devuelve algo fuera de
            // 0-3 (estado transitorio raro, star power interactuando con
            // el byte, etc.) preferimos no escribir fuera del arreglo en
            // vez de tirar una excepcion que corte el episodio entero.
            var clampedPowerup = Math.Clamp(state.PowerupLevel, 0, MaxKnownPowerupLevel);
            for (var level = 0; level < PowerupSignalCount; level++)
            {
                input[powerupInputBase + level] = level == clampedPowerup ? 1f : 0f;
            }

            var nearestSprites = NearestSprites(state, TrackedSpriteCount);
            var spriteInputBase = powerupInputBase + PowerupSignalCount;

            for (var slot = 0; slot < TrackedSpriteCount; slot++)
            {
                var offset = spriteInputBase + slot * SignalsPerSprite;

                if (slot < nearestSprites.Count)
                {
                    var (dx, dy, type) = nearestSprites[slot];
                    input[offset] = dx / SpriteOffsetScale;
                    input[offset + 1] = dy / SpriteOffsetScale;
                    input[offset + 2] = type / SpriteTypeScale;
                }
                else
                {
                    // No hay sprite en este slot (por ejemplo, solo hay 1 enemigo
                    // en pantalla): lo dejamos en 0, que es indistinguible de
                    // "no hay nada ahi" para la red.
                    input[offset] = 0f;
                    input[offset + 1] = 0f;
                    input[offset + 2] = 0f;
                }
            }

            return input;
        }

        private static List<(float Dx, float Dy, float Type)> NearestSprites(SnesState state, int count)
        {
            if (state.Sprites.Count == 0)
            {
                return new List<(float, float, float)>();
            }

            return state.Sprites
                .Select(sprite =>
                {
                    var dx = (float)(sprite.X - state.MarioX);
                    var dy = (float)(sprite.Y - state.MarioY);
                    return (Dx: dx, Dy: dy, Type: (float)sprite.Type, DistanceSquared: dx * dx + dy * dy);
                })
                .OrderBy(candidate => candidate.DistanceSquared)
                .Take(count)
                .Select(candidate => (candidate.Dx, candidate.Dy, candidate.Type))
                .ToList();
        }
    }
}
