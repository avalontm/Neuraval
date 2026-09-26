using Neuraval.Samples.DinoGame.Sources;
using Neuraval.Samples.DinoGame.Sources.UI;
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

        /// <summary>
        /// Muestra/oculta el panel de "como piensa el Dino" (entradas,
        /// activaciones ocultas y salida en tiempo real). Activado por
        /// defecto; se alterna con F2 igual que F1 alterna IsDebug.
        /// </summary>
        bool showBrainViz = true;
        SpriteBatch _spriteBatch;
        StageBackground background;
        SpriteFont font;

        GameState state = GameState.MainMenu;
        GameState stateBeforeOptions = GameState.MainMenu;
        Menu mainMenu;
        Menu optionsMenu;
        Menu pauseMenu;

        // Todos los parametros de poblacion/evolucion/dificultad se cargan
        // desde JSON (se crea con valores por defecto la primera vez que se
        // ejecuta el juego). Ver DinoTrainingSettings para el detalle y la
        // ruta del archivo.
        readonly DinoTrainingSettings settings;

        readonly Random evolutionRandom = new Random();
        readonly ElitistMutationStrategy<NeuralNetwork> evolutionStrategy;

        float bestFitnessEver = 0f;
        DinoGenome bestGenomeEver = null;

        ProbabilidadPorcentaje probabilidad;
        int every_sec = 0;
        int generation = 0;
        int alive = 0;
        Dino playerTarget;

        // Tiempo transcurrido desde que empezo la ronda/generacion actual
        // (se reinicia en GameStart). Antes la rampa de velocidad usaba
        // gameTime.TotalGameTime, que es el tiempo total desde que arranco
        // la aplicacion: eso hacia que la dificultad se acelerase mas y mas
        // rapido en cada generacion sucesiva de una misma sesion larga, sin
        // relacion alguna con el desempeño de esa ronda, y volvia el fitness
        // no comparable entre generaciones (la base misma del algoritmo
        // genetico). Con este contador propio, cada ronda arranca con la
        // misma rampa de dificultad que todas las demas.
        double roundElapsedSeconds = 0d;

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

            settings = DinoTrainingSettings.Load();
            speedStart = settings.SpeedStart;
            DinoStartX = settings.DinoStartX;
            DinoStartXJitter = settings.DinoStartXJitter;

            evolutionStrategy = new ElitistMutationStrategy<NeuralNetwork>(
                evolutionRandom, settings.EliteCount, settings.MutationRate, settings.MutationStrength,
                settings.RandomInjectionFraction, () => new NeuralNetwork());
        }

        /// <summary>
        /// Posicion X base y variacion aleatoria (+/-) donde arranca cada
        /// dinosaurio al reiniciar (<see cref="Dino.Reset"/> las lee de
        /// aqui, igual que ya leia <see cref="speed"/>). Vienen de
        /// <see cref="DinoTrainingSettings"/>.
        /// </summary>
        public static int DinoStartX = 200;
        public static int DinoStartXJitter = 80;

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
                    savedData, settings.PopulationSize, evolutionRandom, settings.MutationRate, settings.MutationStrength,
                    settings.RandomInjectionFraction);

                foreach (var brain in brains)
                {
                    players.Add(new Dino(brain));
                }
            }
            else
            {
                for (int i = 0; i < settings.PopulationSize; i++)
                {
                    players.Add(new Dino());
                }
            }

            alive = players.Count;

            BuildMenus();
        }

        void BuildMenus()
        {
            mainMenu = new Menu(new[]
            {
                new MenuItem("Nuevo Juego", StartGame),
                new MenuItem("Opciones", OpenOptions),
                new MenuItem("Salir", () => Exit())
            });

            optionsMenu = new Menu(new[]
            {
                new MenuItem(() => $"Pantalla completa: {(_graphics.IsFullScreen ? "ON" : "OFF")}", ToggleFullscreen),
                new MenuItem(() => $"HUD de depuracion: {(IsDebug ? "ON" : "OFF")}", ToggleDebugHud),
                new MenuItem(() => $"Red neuronal: {(showBrainViz ? "ON" : "OFF")}", ToggleBrainViz),
                new MenuItem("Volver", CloseOptions)
            });

            pauseMenu = new Menu(new[]
            {
                new MenuItem("Continuar", ResumeGame),
                new MenuItem("Opciones", OpenOptions),
                new MenuItem("Menu Principal", GoToMainMenu),
                new MenuItem("Salir", () => Exit())
            });
        }

        void StartGame()
        {
            state = GameState.Playing;
        }

        void OpenOptions()
        {
            stateBeforeOptions = state;
            optionsMenu.Reset();
            state = GameState.Options;
        }

        void CloseOptions()
        {
            state = stateBeforeOptions;
        }

        void PauseGame()
        {
            pauseMenu.Reset();
            state = GameState.Paused;
        }

        void ResumeGame()
        {
            state = GameState.Playing;
        }

        void GoToMainMenu()
        {
            mainMenu.Reset();
            state = GameState.MainMenu;
        }

        void ToggleFullscreen()
        {
            _graphics.IsFullScreen = !_graphics.IsFullScreen;
            _graphics.ApplyChanges();
        }

        void ToggleDebugHud()
        {
            IsDebug = !IsDebug;
        }

        void ToggleBrainViz()
        {
            showBrainViz = !showBrainViz;
        }

        protected override void Update(GameTime gameTime)
        {
            base.Update(gameTime);
            time = gameTime;

            InputManager.GetState();

            switch (state)
            {
                case GameState.MainMenu:
                    mainMenu.Update();
                    break;

                case GameState.Options:
                    optionsMenu.Update();
                    if (InputManager.IsKeyPressed(Keys.Escape, true))
                    {
                        CloseOptions();
                    }
                    break;

                case GameState.Paused:
                    pauseMenu.Update();
                    if (InputManager.IsKeyPressed(Keys.Escape, true))
                    {
                        ResumeGame();
                    }
                    break;

                case GameState.Playing:
                    UpdateGameplay(gameTime);
                    break;
            }
        }

        void UpdateGameplay(GameTime gameTime)
        {
            if (GameOver)
            {
                if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed
                    || Keyboard.GetState().IsKeyDown(Keys.Escape)
                    || Keyboard.GetState().IsKeyDown(Keys.Enter)
                    || Keyboard.GetState().IsKeyDown(Keys.Space))
                {
                    GameStart();
                }

                return;
            }

            if (InputManager.IsKeyPressed(Keys.Escape, true))
            {
                PauseGame();
                return;
            }

            if (InputManager.IsKeyPressed( Keys.F1, true))
            {
                IsDebug = !IsDebug;
            }

            if (InputManager.IsKeyPressed(Keys.F2, true))
            {
                showBrainViz = !showBrainViz;
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

                // Se itera hacia atras porque BaseEnemy.Update puede
                // eliminarse a si mismo de la lista (cuando sale de
                // pantalla). Iterando hacia adelante, una eliminacion
                // desplaza los indices siguientes y hace que el elemento
                // que ocupa el hueco se salte su Update ese frame; iterando
                // hacia atras, una eliminacion en c nunca afecta a los
                // indices 0..c-1 que faltan por procesar.
                for (int c = enemies.Count - 1; c >= 0; c--)
                {
                    enemies[c].Update(speed);
                }

                if (every_sec > settings.EnemySpawnIntervalFrames)
                {
                    every_sec = 0;
                    Spawn_Enemy();
                }

                every_sec += 1;
                roundElapsedSeconds += gameTime.ElapsedGameTime.TotalSeconds;
                speed += (float)(settings.SpeedRampPerSecond * roundElapsedSeconds);
            }
        }

        void GameStart()
        {
            enemies.Clear();

            EvolvePopulation();

            generation++;
            every_sec = 0;
            speed = speedStart;
            roundElapsedSeconds = 0d;
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

            // Solo hace falta exportar (copiar todos los pesos de) los
            // genomas que realmente se van a guardar en disco: el mejor y,
            // como mucho, MaxEliteGenomesToSave. Antes se llamaba
            // ExportGenome() para los 1000 dinos de la poblacion en cada
            // generacion aunque el 94% se descartara enseguida; ordenar
            // primero por fitness (sin exportar nada) y exportar solo el
            // top-K es mucho mas barato para poblaciones grandes.
            var rankedPlayers = players.OrderByDescending(p => p.Fitness).ToList();
            int eliteToExport = Math.Min(settings.MaxEliteGenomesToSave, rankedPlayers.Count);
            var ranked = rankedPlayers
                .Take(eliteToExport)
                .Select(p => (Genome: p.Brain.ExportGenome(), Fitness: p.Fitness))
                .ToList();

            var nextGenerationBrains = evolutionStrategy.NextGeneration(brains, fitnessScores).ToList();

            // El elitismo de evolutionStrategy solo mira el top de ESTA
            // ronda: si el mejor genoma de toda la sesion tuvo mala suerte
            // (obstaculos dificiles) y esta ronda no lo igualo ni lo
            // supero, puede quedar fuera del top y perderse de la
            // poblacion en vivo para siempre. Antes solo se garantizaba
            // que sobreviviera al recargar desde disco (ver
            // DinoEvolutionStore.RebuildPopulation); esto hace que la
            // misma garantia aplique tambien dentro de una sesion en vivo,
            // reinyectandolo sin mutar en el ultimo slot de la siguiente
            // generacion (parte del cupo de "sangre nueva", el que menos
            // probablemente ya fuera valioso).
            float bestThisRound = rankedPlayers.Count > 0 ? rankedPlayers[0].Fitness : 0f;
            if (bestGenomeEver != null && bestThisRound < bestFitnessEver && nextGenerationBrains.Count > 0)
            {
                nextGenerationBrains[nextGenerationBrains.Count - 1] = NeuralNetwork.FromGenome(bestGenomeEver);
            }

            for (int i = 0; i < players.Count; i++)
            {
                players[i].SetBrain(nextGenerationBrains[i]);
                players[i].Reset();
            }

            // Guarda el progreso en disco despues de cada generacion para
            // que, si se cierra el juego, la proxima vez arranque desde
            // aqui en lugar de perder todo lo aprendido.
            var saveData = DinoEvolutionStore.BuildSaveData(ranked, generation + 1, bestFitnessEver, bestGenomeEver, settings.MaxEliteGenomesToSave);
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
            bool menuBackdrop = state == GameState.MainMenu || state == GameState.Options;
            GraphicsDevice.Clear(menuBackdrop ? Color.Black : Color.White);
            _spriteBatch.Begin();

            switch (state)
            {
                case GameState.MainMenu:
                    DrawMainMenuScreen();
                    break;

                case GameState.Options:
                    DrawOptionsScreen();
                    break;

                case GameState.Paused:
                    DrawGameplay();
                    DrawPauseScreen();
                    break;

                case GameState.Playing:
                    DrawGameplay();
                    break;
            }

            _spriteBatch.End();
            base.Draw(gameTime);
        }

        void DrawGameplay()
        {
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
            DrawBrainVisualizer();

            if (GameOver)
            {
                DrawGameOverScreen();
            }
        }

        float CenterX(string text, float scale)
        {
            return (_graphics.PreferredBackBufferWidth - font.MeasureString(text).X * scale) / 2f;
        }

        void DrawMainMenuScreen()
        {
            int screenW = _graphics.PreferredBackBufferWidth;
            int screenH = _graphics.PreferredBackBufferHeight;

            string title = "NEURAVAL DINO";
            string subtitle = "Entrenamiento evolutivo NEAT";
            float titleScale = 2.4f;
            float subtitleScale = 1f;

            Vector2 titlePos = new Vector2(CenterX(title, titleScale), screenH * 0.22f);
            Vector2 subtitlePos = new Vector2(CenterX(subtitle, subtitleScale), titlePos.Y + font.MeasureString(title).Y * titleScale + 12);

            _spriteBatch.DrawString(font, title, titlePos, Color.Gold, 0f, Vector2.Zero, titleScale, SpriteEffects.None, 0f);
            _spriteBatch.DrawString(font, subtitle, subtitlePos, Color.White, 0f, Vector2.Zero, subtitleScale, SpriteEffects.None, 0f);

            Vector2 menuOrigin = new Vector2(screenW / 2f - 100, screenH * 0.5f);
            mainMenu.Draw(_spriteBatch, font, menuOrigin, 40f, 1.1f);

            string hint = "Flechas para moverte  -  Enter para elegir";
            Vector2 hintPos = new Vector2(CenterX(hint, 0.7f), screenH - 40);
            _spriteBatch.DrawString(font, hint, hintPos, new Color(200, 200, 200), 0f, Vector2.Zero, 0.7f, SpriteEffects.None, 0f);
        }

        void DrawOptionsScreen()
        {
            int screenW = _graphics.PreferredBackBufferWidth;
            int screenH = _graphics.PreferredBackBufferHeight;

            string title = "OPCIONES";
            Vector2 titlePos = new Vector2(CenterX(title, 1.8f), screenH * 0.2f);
            _spriteBatch.DrawString(font, title, titlePos, Color.Gold, 0f, Vector2.Zero, 1.8f, SpriteEffects.None, 0f);

            Vector2 menuOrigin = new Vector2(screenW / 2f - 180, screenH * 0.42f);
            optionsMenu.Draw(_spriteBatch, font, menuOrigin, 40f, 1f);

            string hint = "Enter para cambiar  -  Esc para volver";
            Vector2 hintPos = new Vector2(CenterX(hint, 0.7f), screenH - 40);
            _spriteBatch.DrawString(font, hint, hintPos, new Color(200, 200, 200), 0f, Vector2.Zero, 0.7f, SpriteEffects.None, 0f);
        }

        void DrawPauseScreen()
        {
            int screenW = _graphics.PreferredBackBufferWidth;
            int screenH = _graphics.PreferredBackBufferHeight;

            DrawManager.DrawLine(_spriteBatch, new Rectangle(0, 0, screenW, screenH), new Color(0, 0, 0, 170));

            string title = "PAUSA";
            Vector2 titlePos = new Vector2(CenterX(title, 2f), screenH * 0.25f);
            _spriteBatch.DrawString(font, title, titlePos, Color.Gold, 0f, Vector2.Zero, 2f, SpriteEffects.None, 0f);

            Vector2 menuOrigin = new Vector2(screenW / 2f - 100, screenH * 0.45f);
            pauseMenu.Draw(_spriteBatch, font, menuOrigin, 40f, 1.1f);
        }

        void Spawn_Enemy()
        {
            if (probabilidad.GenerarConProbabilidad(settings.BirdSpawnProbabilityPercent))
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

            _spriteBatch.DrawString(font, $"Generacion: {generation}", new Vector2(10, 20), Color.Black);
            _spriteBatch.DrawString(font, $"Vivos: {alive}", new Vector2(10, 40), Color.Black);
            _spriteBatch.DrawString(font, $"Mejor fitness historico: {bestFitnessEver:0}", new Vector2(10, 60), Color.Black);
            _spriteBatch.DrawString(font, "[F2] Mostrar/ocultar red neuronal  -  [Esc] Pausa", new Vector2(10, 80), Color.Black, 0f, Vector2.Zero, 0.8f, SpriteEffects.None, 0f);
        }

        /// <summary>
        /// Dibuja el panel de "como piensa el Dino" para el mismo
        /// dinosaurio que ya se usa para el resto del HUD de depuracion
        /// (<see cref="playerTarget"/>, fijado en <see cref="DrawDebug"/>).
        /// Solo se pide el snapshot a un dinosaurio por fotograma (no a los
        /// hasta miles de la poblacion), asi que el costo extra es mínimo.
        /// </summary>
        void DrawBrainVisualizer()
        {
            if (!showBrainViz || GameOver || playerTarget == null || playerTarget.dead)
            {
                return;
            }

            var snapshot = playerTarget.CaptureBrainSnapshot();

            int panelWidth = 360;
            int panelHeight = 420;
            var panelRect = new Rectangle(
                _graphics.PreferredBackBufferWidth - panelWidth - 20,
                70,
                panelWidth,
                panelHeight);

            NeuralNetworkVisualizer.Draw(_spriteBatch, font, snapshot, panelRect);
        }
    }
}
