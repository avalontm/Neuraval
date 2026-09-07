using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using System;
using System.Linq;

namespace Neuraval.Samples.DinoGame.Sources
{

    public class Dino
    {
        public int x { set; get; }
        public int y { set; get; }
        public int w { set; get; }
        public int h { set; get; }
        public bool jumping { set; get; }
        public bool crounching { set; get; }
        public float jump_stage { set; get; }
        public bool dead { set; get; }

        //Colision
        public Rectangle Bounds { private set; get; }

        //Red Reunoral (Inteligencia Artificial)
        NeuralNetwork neuralNetwork;

        /// <summary>
        /// Cerebro actual de este dinosaurio. Se expone para poder leerlo
        /// (evaluar/guardar el genoma) y reemplazarlo (<see cref="SetBrain"/>)
        /// cuando el algoritmo genetico construye la siguiente generacion.
        /// </summary>
        public NeuralNetwork Brain => neuralNetwork;

        /// <summary>
        /// Puntuacion de aptitud (fitness) de este dinosaurio en la partida
        /// actual. Aumenta mientras esta vivo en funcion de la distancia
        /// recorrida (velocidad * tiempo), asi que premia tanto sobrevivir
        /// mucho tiempo como sobrevivir a velocidades mas altas. Deja de
        /// crecer en cuanto muere, por lo que su valor final es la nota que
        /// usa el algoritmo genetico para seleccionar a los mejores.
        /// </summary>
        public float Fitness { private set; get; }

        //Animacion
        int fotogramaActual;
        float tiempoTranscurrido;
        float tiempoCambioFotograma = 0.1f; // Cambia el fotograma cada 0.1 segundos

        public Dino()
        {
            neuralNetwork = new NeuralNetwork();

            Reset();

        }

        /// <summary>
        /// Crea un dinosaurio con un cerebro ya existente (por ejemplo, uno
        /// producido por el algoritmo genetico o cargado desde disco), en
        /// vez de uno nuevo con pesos aleatorios.
        /// </summary>
        public Dino(NeuralNetwork brain)
        {
            neuralNetwork = brain;

            Reset();
        }

        /// <summary>
        /// Sustituye el cerebro de este dinosaurio, reutilizando la
        /// instancia (y su representacion visual) para la siguiente
        /// generacion en vez de crear objetos nuevos en cada ronda.
        /// </summary>
        public void SetBrain(NeuralNetwork brain)
        {
            neuralNetwork = brain;
        }

        public void Reset()
        {
            Random rnd = new Random();

            x = 200 + rnd.Next(-80, 80);
            y = 450;
            w = 80;
            h = 86;

            jumping = false;
            crounching = false;

            Bounds = new Rectangle(x, y, w, h);
            dead = false;
            Fitness = 0f;
        }

        public void Update(GameTime gameTime)
        {
            if (!dead)
            {
                // Recompensa por seguir vivo: distancia recorrida en este
                // fotograma (velocidad del juego * tiempo transcurrido).
                // Al dejar de actualizarse en cuanto "dead" es true, el
                // ultimo valor acumulado queda como el fitness final.
                Fitness += MainGame.speed * (float)gameTime.ElapsedGameTime.TotalSeconds;

                if (jumping)
                {
                    y = 448 - (int)f(jump_stage);
                    jump_stage += 0.03f;
                    if (jump_stage > 1)
                    {
                        jumping = false;
                        crounching = false;
                        jump_stage = 0;
                        y = 450;
                    }
                }
                else if (crounching)
                {
                    crounching = false;
                    y -= 34;
                    w = 80;
                    h = 86;
                }
                else
                {
                    jumping = false;
                    crounching = false;
                    jump_stage = 0;
                    y = 450;
                }

                // Actualiza el agente de IA
                onIA();

                onCollition();
            }
        }

        public void onIA()
        {
            // Obtener las características del entorno (por ejemplo, la posición del dinosaurio y el cactus)
            float[] input = GetGameStatus();

            // Obtener la predicción de la red neuronal
            float[] output = neuralNetwork.Predict(input);

            // FeedForwardNetwork no tiene una capa de salida acotada a [0,1]
            // (usa activacion lineal en la ultima capa), asi que la decision
            // se toma por el signo de la salida en vez del umbral 0.5 que
            // usaba la version con Accord.Neuro (que si tenia sigmoide).
            float jumpSignal = output[0]; // Esta salida es para saltar
            float duckSignal = output[1]; // Esta salida es para agacharse

            // IMPORTANTE: antes esto eran dos "if" independientes y saltar
            // se evaluaba primero, asi que en cuanto jumpSignal > 0 se
            // fijaba jumping = true y la comprobacion "if (!jumping)" de
            // onDuck() bloqueaba el agachado el resto de ese fotograma -
            // saltar SIEMPRE ganaba sin importar cuanto mas fuerte fuera la
            // señal de agacharse. Ahora se elige una unica accion por
            // fotograma: la de mayor señal (y solo si al menos una supera
            // el umbral de activacion 0); si ninguna lo supera, el dino
            // sigue corriendo con normalidad. Asi la red puede realmente
            // aprender a preferir agacharse cuando le convenga.
            if (jumpSignal <= 0f && duckSignal <= 0f)
            {
                return;
            }

            if (jumpSignal >= duckSignal)
            {
                onJump();
            }
            else
            {
                onDuck();
            }
        }

        public int CalculateDistanceToObstacle()
        {
            int distance = 0; // Inicializar con un valor infinito para encontrar el obstáculo más cercano

            if (MainGame.enemies.Count > 0)
            {
                distance = MainGame.enemies[0].Bounds.X - Bounds.X;
                if (distance < 0)
                {
                    if (MainGame.enemies.Count > 1)
                    {
                        distance = MainGame.enemies[1].Bounds.X - Bounds.X;
                    }
                }
            }

            return distance;
        }

        public int CalculateObstaclePositionX()
        {
            int x = 0;

            if (MainGame.enemies.Count > 0)
            {
                int distance = CalculateDistanceToObstacle();

                if (distance < 0)
                {
                    if (MainGame.enemies.Count > 1)
                    {
                        x = MainGame.enemies[1].Bounds.X;
                    }
                }
                else
                {
                    x = MainGame.enemies[0].Bounds.X;
                }
            }

            return x;
        }

        public int CalculateObstaclePositionY()
        {
            int y = 0;

            if (MainGame.enemies.Count > 0)
            {
                int distance = CalculateDistanceToObstacle();

                if (distance < 0)
                {
                    if (MainGame.enemies.Count > 1)
                    {
                        y = MainGame.enemies[1].Bounds.Y;
                    }
                }
                else
                {
                    y = MainGame.enemies[0].Bounds.Y;
                }
            }

            return y;
        }


        public int CalculateObstacleWidth()
        {
            int w = 0;

            if (MainGame.enemies.Count > 0)
            {
                int distance = CalculateDistanceToObstacle();

                if (distance < 0)
                {
                    if (MainGame.enemies.Count > 1)
                    {
                        w = MainGame.enemies[1].Bounds.Width;
                    }
                }
                else
                {
                    w = MainGame.enemies[0].Bounds.Width;
                }
            }

            return w;
        }

        public int CalculateObstacleHeight()
        {
            int h = 0;

            if (MainGame.enemies.Count > 0)
            {
                int distance = CalculateDistanceToObstacle();

                if (distance < 0)
                {
                    if (MainGame.enemies.Count > 1)
                    {
                        h = MainGame.enemies[1].Bounds.Height;
                    }
                }
                else
                {
                    h = MainGame.enemies[0].Bounds.Height;
                }
            }

            return h;
        }


        float[] GetGameStatus()
        {
            float[] features = new float[7];
            features[0] = CalculateDistanceToObstacle();
            features[1] = CalculateObstaclePositionX();
            features[2] = CalculateObstaclePositionY();
            features[3] = CalculateObstacleWidth();
            features[4] = CalculateObstacleHeight();
            features[5] = Bounds.Y;
            features[6] = MainGame.speed;
            return features;
        }

        void onCollition()
        {
            for (int c = 0; c < MainGame.enemies.Count; c++)
            {
                if (Collition(MainGame.enemies[c]))
                {
                    if (MainGame.players.Where(x => !x.dead).Count() == 1)
                    {
                        MainGame.LastPlayer = this;
                    }

                    dead = true;
                }
            }
        }

        void onDebug(SpriteBatch _spriteBatch)
        {
            DrawManager.DrawLine(_spriteBatch, new Rectangle(Bounds.X, Bounds.Y, Bounds.Width, 1), Color.Red);
            DrawManager.DrawLine(_spriteBatch, new Rectangle(Bounds.X, Bounds.Y, 1, Bounds.Height), Color.Red);
            DrawManager.DrawLine(_spriteBatch, new Rectangle(Bounds.X + Bounds.Width, Bounds.Y, 1, Bounds.Height), Color.Red);
            DrawManager.DrawLine(_spriteBatch, new Rectangle(Bounds.X, Bounds.Y + Bounds.Height, Bounds.Width, 1), Color.Red);
        }

        public void Draw(SpriteBatch _spriteBatch)
        {
            if (!dead)
            {
                int offset = 20;
                Bounds = new Rectangle(x + offset, y + offset, w - (offset*2), h - (offset+5));
                Rectangle rec = new Rectangle(x , y , w , h );
                _spriteBatch.Draw(Animations.dino[0], rec, Color.White);

                if (MainGame.IsDebug)
                {
                    onDebug(_spriteBatch);
                }
            }
        }

        float f(float x)
        {
            return (-4 * x * (x - 1) * 172);
        }

        bool Collition(BaseEnemy enemy)
        {
            return this.Bounds.Intersects(enemy.Bounds);
        }

        public void onJump()
        {
            if (!jumping)
            {
                jumping = true;
                crounching = false;
            }
        }

        public void onDuck()
        {
            if (!jumping)
            {
                jumping = false;
                crounching = true;
                y += 34;
                w = 110;
                h = 52;
            }
        }
    }
}
