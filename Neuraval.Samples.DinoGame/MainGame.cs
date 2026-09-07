using Neuraval.Samples.DinoGame.Sources;
using Neuraval.Evolution;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuraval.Samples.DinoGame
{
    public class MainGame : Game
    {
        public static GraphicsDeviceManager _graphics;
        public static IList<Dino> players;
        public static IList<BaseEnemy> enemies;
        public static GameTime time { private set; get; }
        float speedStart = 12;
        public static float speed = 12;
        public static bool GameOver = false;
        public static Dino LastPlayer;
        public static bool IsDebug { private set; get; }
        SpriteBatch _spriteBatch;
        StageBackground background;
        SpriteFont font;

        const int PopulationSize = 1000;

        // --- Algoritmo genetico ---
        // Cuantos de los mejores dinosaurios de cada generacion pasan sin
        // cambios (elitismo): garantiza que nunca se pierde lo que ya
        // funcionaba bien de una generacion a la siguiente.
        const int EliteCount = 20;
        // Probabilidad de que cada peso/bias individual mute al reproducirse.
        const float MutationRate = 0.12f;
        // Magnitud (desviacion estandar) de la mutacion gaussiana aplicada.
        const float MutationStrength = 0.35f;

        readonly Random evolutionRandom = new Random();
        readonly ElitistMutationStrategy<NeuralNetwork> evolutionStrategy;

        float bestFitnessEver = 0f;
        DinoGenome bestGenomeEver = null;

        ProbabilidadPorcentaje probabilidad;
        int every_sec = 0;
        int generation = 0;
        int alive = 0;
        Dino playerTarget;

        public MainGame()
        {
            _graphics = new GraphicsDeviceManager(this);
            _graphics.PreferredBackBufferHeight = 720;
            _graphics.PreferredBackBufferWidth = 1280;
            _graphics.SynchronizeWithVerticalRetrace = false;

            this.IsFixedTimeStep = true;//false;
            this.TargetElapsedTime = TimeSpan.FromSeconds(1d / 60d);

            Content.RootDirectory = "Content";
            IsMouseVisible = true;

            evolutionStrategy = new ElitistMutationStrategy<NeuralNetwork>(
                evolutionRandom, EliteCount, MutationRate, MutationStrength);
        }

        protected override void Initialize()
        {
            // TODO: Add your initialization logic here

            base.Initialize();
        }

        protected override void LoadContent()
        {
            _spriteBatch = new SpriteBatch(GraphicsDevice);

            // TODO: use this.Content to load your game content here
            font = Content.Load<SpriteFont>("default");

            DrawManager.Init(_graphics.GraphicsDevice);
            probabilidad = new ProbabilidadPorcentaje();
            Animations.Load(Content);

            background = new StageBackground();
            players = new List<Dino>();
            enemies = new List<BaseEnemy>();

            // Intenta continuar la evolucion guardada de una partida
            // anterior (aunque el juego se haya cerrado por completo). Si no
            // hay nada guardado, se arranca con una poblacion totalmente
            // nueva de pesos aleatorios, como antes.
            var savedData = DinoEvolutionStore.Load();

            if (savedData != null)
            {
                generation = savedData.Generation;
                bestFitnessEver = savedData.BestFitnessEver;
                bestGenomeEver = savedData.BestGenomeEver;

                var brains = DinoEvolutionStore.RebuildPopulation(
                    savedData, PopulationSize, evolutionRandom, MutationRate, MutationStrength);

                foreach (var brain in brains)
                {
                    players.Add(new Dino(brain));
                }
            }
            else
            {
                for (int i = 0; i < PopulationSize; i++)
                {
                    players.Add(new Dino());
                }
            }

            alive = players.Count;
        }

        protected override void Update(GameTime gameTime)
        {
            base.Update(gameTime);
            time = gameTime;

            InputManager.GetState();

            if (GameOver)
            {
                if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed
                    || Keyboard.GetState().IsKeyDown(Keys.Escape)
                    || Keyboard.GetState().IsKeyDown(Keys.Enter)
                    || Keyboard.GetState().IsKeyDown(Keys.Space))
                {
                    GameStart();
                }
            }

            if (InputManager.IsKeyPressed( Keys.F1, true))
            {
                IsDebug = !IsDebug;
            }

            onGameOver();

            if (!GameOver)
            {
                // TODO: Add your update logic here
                background.Update(gameTime);

                for (int p = 0; p < players.Count; p++)
                {
                    players[p].Update(gameTime);
                }

                for (int c = 0; c < enemies.Count; c++)
                {
                    enemies[c].Update(speed);
                }

                if (every_sec > 60)
                {
                    every_sec = 0;
                    Spawn_Enemy();
                }

                every_sec += 1;
                speed += (float)(0.00000025f * gameTime.TotalGameTime.TotalSeconds);
            }
        }

        void GameStart()
        {
            enemies.Clear();

            EvolvePopulation();

            generation++;
            every_sec = 0;
            speed = speedStart;
            GameOver = false;
        }

        /// <summary>
        /// Reemplaza la poblacion actual por una nueva generacion obtenida
        /// mediante seleccion + elitismo + mutacion a partir del fitness que
        /// consiguio cada dinosaurio en la ronda que acaba de terminar. Esto
        /// es lo que hace que la poblacion realmente "aprenda" de una ronda
        /// a la siguiente en vez de solo reiniciar posiciones con los mismos
        /// cerebros aleatorios de siempre.
        /// </summary>
        void EvolvePopulation()
        {
            var brains = players.Select(p => p.Brain).ToList();
            var fitnessScores = players.Select(p => p.Fitness).ToList();

            // Genomas ordenados de mejor a peor fitness, usados tanto para
            // decidir el guardado en disco como para depuracion.
            var ranked = players
                .Select(p => (Genome: p.Brain.ExportGenome(), Fitness: p.Fitness))
                .OrderByDescending(pair => pair.Fitness)
                .ToList();

            var nextGenerationBrains = evolutionStrategy.NextGeneration(brains, fitnessScores);

            for (int i = 0; i < players.Count; i++)
            {
                players[i].SetBrain(nextGenerationBrains[i]);
                players[i].Reset();
            }

            // Guarda el progreso en disco despues de cada generacion para
            // que, si se cierra el juego, la proxima vez arranque desde
            // aqui en lugar de perder todo lo aprendido.
            var saveData = DinoEvolutionStore.BuildSaveData(ranked, generation + 1, bestFitnessEver, bestGenomeEver);
            bestFitnessEver = saveData.BestFitnessEver;
            bestGenomeEver = saveData.BestGenomeEver;
            DinoEvolutionStore.Save(saveData);
        }

        void onGameOver()
        {
            alive = players.Where(x => !x.dead).Count();

            if (alive == 0)
            {
                GameOver = true;
            }
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.White);
            _spriteBatch.Begin();

            background.Draw(_spriteBatch);
            // TODO: Add your drawing code here

            for (int p = 0; p < players.Count; p++)
            {
                players[p].Draw(_spriteBatch);
            }

            for (int c = 0; c < enemies.Count; c++)
            {
                enemies[c].Draw(_spriteBatch);
            }

            DrawDebug();

            if (GameOver)
            {
                DrawGameOverScreen();
            }

            _spriteBatch.End();
            base.Draw(gameTime);
        }


        void Spawn_Enemy()
        {
            if (probabilidad.GenerarConProbabilidad(20))
            {
                enemies.Add(new Bird());
            }
            else
            {
                enemies.Add(new Cactus());
            }
        }

        void DrawGameOverScreen()
        {
            int screenW = _graphics.PreferredBackBufferWidth;
            int screenH = _graphics.PreferredBackBufferHeight;

            string title = "GAME OVER";
            string genText = $"Generacion {generation}";
            string hint = "Presiona ENTER, ESPACIO o ESC para continuar";

            Vector2 titleSize = font.MeasureString(title) * 2f;
            Vector2 genSize = font.MeasureString(genText) * 1.2f;
            Vector2 hintSize = font.MeasureString(hint);

            // Fondo semitransparente para que el texto resalte sobre el
            // fondo blanco/escenario del juego.
            int panelWidth = (int)Math.Max(titleSize.X, Math.Max(genSize.X, hintSize.X)) + 80;
            int panelHeight = 180;
            var panelRect = new Rectangle(
                (screenW - panelWidth) / 2,
                (screenH - panelHeight) / 2,
                panelWidth,
                panelHeight);
            DrawManager.DrawLine(_spriteBatch, panelRect, new Color(0, 0, 0, 160));

            Vector2 titlePos = new Vector2((screenW - titleSize.X) / 2f, panelRect.Y + 20);
            Vector2 genPos = new Vector2((screenW - genSize.X) / 2f, titlePos.Y + titleSize.Y + 15);
            Vector2 hintPos = new Vector2((screenW - hintSize.X) / 2f, genPos.Y + genSize.Y + 20);

            _spriteBatch.DrawString(font, title, titlePos, Color.Red, 0f, Vector2.Zero, 2f, SpriteEffects.None, 0f);
            _spriteBatch.DrawString(font, genText, genPos, Color.White, 0f, Vector2.Zero, 1.2f, SpriteEffects.None, 0f);
            _spriteBatch.DrawString(font, hint, hintPos, Color.Yellow);
        }

        void DrawDebug()
        {
            for (int i = 0; i < players.Count; i++)
            {
                if (!players[i].dead)
                {
                    playerTarget = players[i];
                    break;
                }
            }

            _spriteBatch.DrawString(font, $"(Obstaculo) Distancia: {playerTarget.CalculateDistanceToObstacle()}", new Vector2(10, 20), Color.Black);
            _spriteBatch.DrawString(font, $"(Obstaculo) X: {playerTarget.CalculateObstaclePositionX()}", new Vector2(10, 40), Color.Black);
            _spriteBatch.DrawString(font, $"(Obstaculo) Y: {playerTarget.CalculateObstaclePositionY()}", new Vector2(10, 60), Color.Black);
            _spriteBatch.DrawString(font, $"(Obstaculo) Ancho: {playerTarget.CalculateObstacleWidth()}", new Vector2(10, 80), Color.Black);
            _spriteBatch.DrawString(font, $"(Obstaculo) Alto: {playerTarget.CalculateObstacleHeight()}", new Vector2(10, 100), Color.Black);
            _spriteBatch.DrawString(font, $"(Dino) Y: {playerTarget.y}", new Vector2(10, 120), Color.Black);
            _spriteBatch.DrawString(font, $"(Juego) Velocidad: {speed}", new Vector2(10, 140), Color.Black);

            _spriteBatch.DrawString(font, $"Generacion: {generation}", new Vector2(_graphics.PreferredBackBufferWidth - 250, 20), Color.Black);
            _spriteBatch.DrawString(font, $"Vivos: {alive}", new Vector2(_graphics.PreferredBackBufferWidth - 220, 40), Color.Black);
            _spriteBatch.DrawString(font, $"Mejor fitness historico: {bestFitnessEver:0}", new Vector2(_graphics.PreferredBackBufferWidth - 340, 60), Color.Black);
        }
    }
}
