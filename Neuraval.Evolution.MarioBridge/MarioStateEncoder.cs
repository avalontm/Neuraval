namespace Neuraval.Evolution.MarioBridge
{
    public static class MarioStateEncoder
    {
        public const int GridCellCount = SnesState.GridSize * SnesState.GridSize;
        public const int TrackedSpriteCount = 3;
        public const int SignalsPerSprite = 25;
        public const int SpriteFlagSignalCount = 4;
        public const int SpriteMiscSignalCount = 3;
        public const int SpriteEntitySignalCount = 4;
        public const int TrackedClusterCount = 3;
        public const int SignalsPerCluster = 2;
        public const int VelocitySignalCount = 2;
        public const int GroundedSignalCount = 1;
        public const int MarioStateSignalCount = 16;
        public const int MarioMiscSignalCount = 3;
        public const int BlockedRightMask = 0x01;
        public const int BlockedLeftMask = 0x02;
        public const int BlockedDownMask = 0x04;
        public const int BlockedUpMask = 0x08;
        public const int MarioBlockSignalCount = VelocitySignalCount + GroundedSignalCount + MarioStateSignalCount + MarioMiscSignalCount;
        public const int CameraSignalCount = 7;
        public const int GameSignalCount = 10;
        public const int PowerupSignalCount = 4;
        public const int MaxKnownPowerupLevel = PowerupSignalCount - 1;
        public const int SpriteBlockSignalCount = TrackedSpriteCount * SignalsPerSprite;
        public const int ClusterBlockSignalCount = TrackedClusterCount * SignalsPerCluster;
        public const int SecondPlayerBlockSignalCount = 4;
        public const int CoinSignalCount = 6;
        public const int DialogSignalCount = 3;
        public const int CliffSignalCount = 4;

        // Bloque dedicado a la Moneda Yoshi (Dragon Coin) mas cercana, igual de
        // prioritario que el de monedas normales / bloques de moneda. Sin este
        // bloque, una moneda Yoshi solo aparece en la red si por casualidad cae
        // entre los 3 sprites mas cercanos (bloque generico de sprites), lo cual
        // hace que la IA la ignore quando hay enemigos mas cerca. Senales:
        // [0] cuantas monedas Yoshi hay activas cerca, [1]/[2] dx/dy de la mas
        // cercana.
        public const int YoshiCoinSignalCount = 3;

        // Bloque para "sostener/tirar items usables": banderas directas de RAM de
        // si Mario esta cargando algo ahora mismo ($1470 / $148F) mas la posicion
        // del item agarrable (status $09, "Stationary/Carryable" en $14C8) mas
        // cercano, para que la red sepa a donde caminar para levantarlo.
        // [0] CarryingFlag, [1] HoldingObjectFlag, [2] hay item agarrable cerca,
        // [3]/[4] dx/dy del item agarrable mas cercano.
        public const int CarrySignalCount = 5;

        // Bandera de si Mario acaba de activar el punto medio (checkpoint) del
        // nivel, la barra que guarda el progreso para el respawn ($13CE).
        public const int CheckpointSignalCount = 1;

        public const int WallSignalCount = 3;
        public const float WallDistanceScale = SnesState.GridRadius;

        // Bandera de si el tramo actual admite scroll vertical (torre, subida
        // larga), leida de $7E:1412. Permite priorizar Arriba/Abajo sobre
        // Izquierda/Derecha en niveles verticales.
        public const int VerticalLevelSignalCount = 1;

        // Bloque para la tuberia vertical exit-enabled mas cercana ($0137/$0138
        // en el tilemap), igual de prioritario que el de Yoshi Coin. Solo cubre
        // tuberias por ahora: la deteccion de puertas queda pendiente de una
        // fase futura por falta de un ID de Map16 "act as" confirmado para
        // puertas (ver SMW_RAM_Map_IA.md, seccion 38). [0] cuantas tuberias
        // hay cerca, [1]/[2] dx/dy de la mas cercana.
        public const int PipeSignalCount = 3;

        public const int MarioBlockStart = GridCellCount;
        public const int CameraBlockStart = MarioBlockStart + MarioBlockSignalCount;
        public const int GameBlockStart = CameraBlockStart + CameraSignalCount;
        public const int PowerupBlockStart = GameBlockStart + GameSignalCount;
        public const int SpriteBlockStart = PowerupBlockStart + PowerupSignalCount;
        public const int ClusterBlockStart = SpriteBlockStart + SpriteBlockSignalCount;
        public const int SecondPlayerBlockStart = ClusterBlockStart + ClusterBlockSignalCount;
        public const int CoinBlockStart = SecondPlayerBlockStart + SecondPlayerBlockSignalCount;
        public const int DialogBlockStart = CoinBlockStart + CoinSignalCount;
        public const int CliffBlockStart = DialogBlockStart + DialogSignalCount;
        public const int YoshiCoinBlockStart = CliffBlockStart + CliffSignalCount;
        public const int CarryBlockStart = YoshiCoinBlockStart + YoshiCoinSignalCount;
        public const int CheckpointBlockStart = CarryBlockStart + CarrySignalCount;
        public const int WallBlockStart = CheckpointBlockStart + CheckpointSignalCount;
        public const int VerticalLevelBlockStart = WallBlockStart + WallSignalCount;
        public const int PipeBlockStart = VerticalLevelBlockStart + VerticalLevelSignalCount;
        public const int InputCount = PipeBlockStart + PipeSignalCount;

        public const float VelocityScale = 16f;
        public const float SpriteOffsetScale = 100f;
        public const float SpriteTypeScale = 255f;
        public const float StatusScale = 255f;
        public const float SubpixelScale = 255f;
        public const float AirStateScale = 255f;
        public const float PMeterScale = 80f;
        public const float MeterScale = 255f;
        public const float CoinsScale = 255f;
        public const float CoinCountScale = 16f;
        public const float CoinOffsetScale = 192f;
        public const float ScreenXScale = 256f;
        public const float ScreenYScale = 240f;
        public const float AbsoluteXScale = 4096f;

        public static float[] Encode(SnesState state)
        {
            var input = new float[InputCount];

            for (var i = 0; i < GridCellCount; i++)
            {
                input[i] = state.Tiles[i] / 255f;
            }

            var m = MarioBlockStart;
            input[m + 0] = state.MarioVelocityX / VelocityScale;
            input[m + 1] = state.MarioVelocityY / VelocityScale;
            input[m + 2] = state.IsGrounded ? 1f : 0f;
            input[m + 3] = state.Direction;
            input[m + 4] = (state.Blocked & BlockedLeftMask) != 0 ? 1f : 0f;
            input[m + 5] = (state.Blocked & BlockedRightMask) != 0 ? 1f : 0f;
            input[m + 6] = (state.Blocked & BlockedUpMask) != 0 ? 1f : 0f;
            input[m + 7] = (state.Blocked & BlockedDownMask) != 0 ? 1f : 0f;
            input[m + 8] = state.SubPixelX / SubpixelScale;
            input[m + 9] = state.SubPixelY / SubpixelScale;
            input[m + 10] = state.MarioSubSpeed / SubpixelScale;
            input[m + 11] = state.AirState / AirStateScale;
            input[m + 12] = state.Ducking;
            input[m + 13] = state.Climbing;
            input[m + 14] = state.Water;
            input[m + 15] = state.PMeter / PMeterScale;
            input[m + 16] = state.TakeoffMeter / MeterScale;
            input[m + 17] = state.HurtTimer / MeterScale;
            input[m + 18] = state.CapeTimer / MeterScale;
            input[m + 19] = state.Coins / CoinsScale;
            input[m + 20] = state.ItemBox / CoinsScale;
            input[m + 21] = state.ReservedItemBox / CoinsScale;

            var c = CameraBlockStart;
            input[c + 0] = (state.MarioX - state.CameraX) / ScreenXScale;
            input[c + 1] = (state.MarioY - state.CameraY) / ScreenYScale;
            input[c + 2] = state.MarioX / AbsoluteXScale;
            input[c + 3] = (state.Layer2X - state.CameraX) / ScreenXScale;
            input[c + 4] = (state.Layer2Y - state.CameraY) / ScreenYScale;
            input[c + 5] = (state.Layer3X - state.CameraX) / ScreenXScale;
            input[c + 6] = (state.Layer3Y - state.CameraY) / ScreenYScale;

            var g = GameBlockStart;
            input[g + 0] = state.GameMode / 255f;
            input[g + 1] = state.LevelMode / 255f;
            input[g + 2] = state.Translevel / 255f;
            input[g + 3] = state.BluePowTimer / MeterScale;
            input[g + 4] = state.SilverPowTimer / MeterScale;
            input[g + 5] = state.DoorExitCounter / MeterScale;
            input[g + 6] = state.ItemMemory / MeterScale;
            input[g + 7] = state.CurrentPlayer / MeterScale;
            input[g + 8] = state.Character / MeterScale;
            input[g + 9] = state.CurrentPlayerCoins / MeterScale;

            var p = PowerupBlockStart;
            var clampedPowerup = Math.Clamp(state.PowerupLevel, 0, MaxKnownPowerupLevel);
            for (var level = 0; level < PowerupSignalCount; level++)
            {
                input[p + level] = level == clampedPowerup ? 1f : 0f;
            }

            var nearestSprites = NearestSprites(state, TrackedSpriteCount);
            var s = SpriteBlockStart;
            for (var slot = 0; slot < TrackedSpriteCount; slot++)
            {
                var offset = s + slot * SignalsPerSprite;

                if (slot < nearestSprites.Count)
                {
                    var sprite = nearestSprites[slot];
                    input[offset + 0] = sprite.Dx / SpriteOffsetScale;
                    input[offset + 1] = sprite.Dy / SpriteOffsetScale;
                    input[offset + 2] = sprite.Type / SpriteTypeScale;
                    input[offset + 3] = sprite.Vx / VelocityScale;
                    input[offset + 4] = sprite.Vy / VelocityScale;
                    input[offset + 5] = sprite.Direction;
                    input[offset + 6] = (sprite.Blocked & BlockedLeftMask) != 0 ? 1f : 0f;
                    input[offset + 7] = (sprite.Blocked & BlockedRightMask) != 0 ? 1f : 0f;
                    input[offset + 8] = (sprite.Blocked & BlockedUpMask) != 0 ? 1f : 0f;
                    input[offset + 9] = (sprite.Blocked & BlockedDownMask) != 0 ? 1f : 0f;
                    input[offset + 10] = sprite.SubX / SubpixelScale;
                    input[offset + 11] = sprite.SubY / SubpixelScale;
                    input[offset + 12] = sprite.Status / StatusScale;
                    input[offset + 13] = sprite.StunTimer / MeterScale;
                    input[offset + 14] = (sprite.Properties & 0x10) != 0 ? 1f : 0f;
                    input[offset + 15] = (sprite.Properties & 0x20) != 0 ? 1f : 0f;
                    input[offset + 16] = (sprite.Properties & 0x02) != 0 ? 1f : 0f;
                    input[offset + 17] = (sprite.Properties & 0x04) != 0 ? 1f : 0f;
                    input[offset + 18] = sprite.Misc1 / MeterScale;
                    input[offset + 19] = sprite.Misc2 / MeterScale;
                    input[offset + 20] = sprite.Misc3 / MeterScale;
                    input[offset + 21] = sprite.OffscreenFull / MeterScale;
                    input[offset + 22] = sprite.Eaten / MeterScale;
                    input[offset + 23] = sprite.ObjectInteraction / MeterScale;
                    input[offset + 24] = sprite.SpinTimer / MeterScale;
                }
            }

            var nearestClusters = NearestClusters(state, TrackedClusterCount);
            var k = ClusterBlockStart;
            for (var slot = 0; slot < TrackedClusterCount; slot++)
            {
                var offset = k + slot * SignalsPerCluster;

                if (slot < nearestClusters.Count)
                {
                    var cluster = nearestClusters[slot];
                    input[offset + 0] = cluster.Dx / SpriteOffsetScale;
                    input[offset + 1] = cluster.Dy / SpriteOffsetScale;
                }
            }

            var t = SecondPlayerBlockStart;
            input[t + 0] = state.P2Controller1 / StatusScale;
            input[t + 1] = state.P2Controller1Prev / StatusScale;
            input[t + 2] = state.P2Controller2 / StatusScale;
            input[t + 3] = state.P2Controller2Prev / StatusScale;

            var cn = CoinBlockStart;
            input[cn + 0] = state.CoinsNear / CoinCountScale;
            input[cn + 1] = state.NearestCoinDx / CoinOffsetScale;
            input[cn + 2] = state.NearestCoinDy / CoinOffsetScale;
            input[cn + 3] = state.CoinBlocksNear / CoinCountScale;
            input[cn + 4] = state.NearestCoinBlockDx / CoinOffsetScale;
            input[cn + 5] = state.NearestCoinBlockDy / CoinOffsetScale;

            var dl = DialogBlockStart;
            input[dl + 0] = state.DialogNear / CoinCountScale;
            input[dl + 1] = state.NearestDialogDx / CoinOffsetScale;
            input[dl + 2] = state.NearestDialogDy / CoinOffsetScale;

            var cl = CliffBlockStart;
            for (var i = 0; i < 2 && i < state.CliffGaps.Count; i++)
            {
                input[cl + i * 2] = state.CliffGaps[i].StartTiles / 7f;
                input[cl + i * 2 + 1] = state.CliffGaps[i].WidthTiles / 8f;
            }

            var yc = YoshiCoinBlockStart;
            var (yoshiCoinCount, yoshiCoinDx, yoshiCoinDy) = NearestByType(state, MarioSpriteNames.IsYoshiCoin);
            input[yc + 0] = yoshiCoinCount / CoinCountScale;
            input[yc + 1] = yoshiCoinDx / SpriteOffsetScale;
            input[yc + 2] = yoshiCoinDy / SpriteOffsetScale;

            var ca = CarryBlockStart;
            var (carryableCount, carryableDx, carryableDy) = NearestByStatus(state, MarioSpriteStatus.StationaryCarryable);
            input[ca + 0] = state.CarryingFlag != 0 ? 1f : 0f;
            input[ca + 1] = state.HoldingObjectFlag != 0 ? 1f : 0f;
            input[ca + 2] = carryableCount > 0 ? 1f : 0f;
            input[ca + 3] = carryableDx / SpriteOffsetScale;
            input[ca + 4] = carryableDy / SpriteOffsetScale;

            var chk = CheckpointBlockStart;
            input[chk + 0] = state.MidwayPointReached ? 1f : 0f;

            var w = WallBlockStart;
            input[w + 0] = state.WallAheadDistance / WallDistanceScale;
            input[w + 1] = state.SolidAboveDistance / WallDistanceScale;
            input[w + 2] = state.SolidBelowDistance / WallDistanceScale;

            var vl = VerticalLevelBlockStart;
            input[vl + 0] = state.IsVerticalLevel ? 1f : 0f;

            var pp = PipeBlockStart;
            input[pp + 0] = state.PipeNear / CoinCountScale;
            input[pp + 1] = state.NearestPipeDx / SpriteOffsetScale;
            input[pp + 2] = state.NearestPipeDy / SpriteOffsetScale;

            return input;
        }

        private static (int Count, float Dx, float Dy) NearestByType(SnesState state, Func<int, bool> matches)
        {
            var count = 0;
            var bestDistanceSquared = float.MaxValue;
            var bestDx = 0f;
            var bestDy = 0f;

            foreach (var sprite in state.Sprites)
            {
                if (!matches(sprite.Type))
                {
                    continue;
                }

                count++;
                var dx = (float)(sprite.X - state.MarioX);
                var dy = (float)(sprite.Y - state.MarioY);
                var distanceSquared = dx * dx + dy * dy;
                if (distanceSquared < bestDistanceSquared)
                {
                    bestDistanceSquared = distanceSquared;
                    bestDx = dx;
                    bestDy = dy;
                }
            }

            return (count, bestDx, bestDy);
        }

        private static (int Count, float Dx, float Dy) NearestByStatus(SnesState state, int status)
        {
            return NearestByPredicate(state, sprite => sprite.Status == status);
        }

        private static (int Count, float Dx, float Dy) NearestByPredicate(SnesState state, Func<SnesSprite, bool> matches)
        {
            var count = 0;
            var bestDistanceSquared = float.MaxValue;
            var bestDx = 0f;
            var bestDy = 0f;

            foreach (var sprite in state.Sprites)
            {
                if (!matches(sprite))
                {
                    continue;
                }

                count++;
                var dx = (float)(sprite.X - state.MarioX);
                var dy = (float)(sprite.Y - state.MarioY);
                var distanceSquared = dx * dx + dy * dy;
                if (distanceSquared < bestDistanceSquared)
                {
                    bestDistanceSquared = distanceSquared;
                    bestDx = dx;
                    bestDy = dy;
                }
            }

            return (count, bestDx, bestDy);
        }

        private static List<SpriteFeature> NearestSprites(SnesState state, int count)
        {
            if (state.Sprites.Count == 0)
            {
                return new List<SpriteFeature>();
            }

            return state.Sprites
                .Select(sprite =>
                {
                    var dx = (float)(sprite.X - state.MarioX);
                    var dy = (float)(sprite.Y - state.MarioY);
                    return new SpriteFeature(
                        dx,
                        dy,
                        sprite.Type,
                        sprite.VelocityX,
                        sprite.VelocityY,
                        sprite.Direction,
                        sprite.Blocked,
                        sprite.SubPixelX,
                        sprite.SubPixelY,
                        sprite.Status,
                        sprite.StunTimer,
                        sprite.Properties,
                        sprite.Misc1,
                        sprite.Misc2,
                        sprite.Misc3,
                        sprite.OffscreenFull,
                        sprite.Eaten,
                        sprite.ObjectInteraction,
                        sprite.SpinTimer,
                        dx * dx + dy * dy);
                })
                .OrderBy(candidate => candidate.DistanceSquared)
                .Take(count)
                .Select(candidate => new SpriteFeature(
                    candidate.Dx,
                    candidate.Dy,
                    candidate.Type,
                    candidate.Vx,
                    candidate.Vy,
                    candidate.Direction,
                    candidate.Blocked,
                    candidate.SubX,
                    candidate.SubY,
                    candidate.Status,
                    candidate.StunTimer,
                    candidate.Properties,
                    candidate.Misc1,
                    candidate.Misc2,
                    candidate.Misc3,
                    candidate.OffscreenFull,
                    candidate.Eaten,
                    candidate.ObjectInteraction,
                    candidate.SpinTimer,
                    0f))
                .ToList();
        }

        private static List<ClusterFeature> NearestClusters(SnesState state, int count)
        {
            if (state.ClusterSprites is null || state.ClusterSprites.Count == 0)
            {
                return new List<ClusterFeature>();
            }

            return state.ClusterSprites
                .Select(cluster =>
                {
                    var dx = (float)(cluster.X - state.MarioX);
                    var dy = (float)(cluster.Y - state.MarioY);
                    return new ClusterFeature(dx, dy, dx * dx + dy * dy);
                })
                .OrderBy(candidate => candidate.DistanceSquared)
                .Take(count)
                .Select(candidate => new ClusterFeature(candidate.Dx, candidate.Dy, 0f))
                .ToList();
        }

        private readonly struct SpriteFeature
        {
            public float Dx { get; }
            public float Dy { get; }
            public float Type { get; }
            public float Vx { get; }
            public float Vy { get; }
            public float Direction { get; }
            public int Blocked { get; }
            public float SubX { get; }
            public float SubY { get; }
            public float Status { get; }
            public float StunTimer { get; }
            public int Properties { get; }
            public float Misc1 { get; }
            public float Misc2 { get; }
            public float Misc3 { get; }
            public float OffscreenFull { get; }
            public float Eaten { get; }
            public float ObjectInteraction { get; }
            public float SpinTimer { get; }
            public float DistanceSquared { get; }

            public SpriteFeature(float dx, float dy, float type, float vx, float vy, float direction, int blocked, float subX, float subY, float status, float stunTimer, int properties, float misc1, float misc2, float misc3, float offscreenFull, float eaten, float objectInteraction, float spinTimer, float distanceSquared)
            {
                Dx = dx;
                Dy = dy;
                Type = type;
                Vx = vx;
                Vy = vy;
                Direction = direction;
                Blocked = blocked;
                SubX = subX;
                SubY = subY;
                Status = status;
                StunTimer = stunTimer;
                Properties = properties;
                Misc1 = misc1;
                Misc2 = misc2;
                Misc3 = misc3;
                OffscreenFull = offscreenFull;
                Eaten = eaten;
                ObjectInteraction = objectInteraction;
                SpinTimer = spinTimer;
                DistanceSquared = distanceSquared;
            }
        }

        private readonly struct ClusterFeature
        {
            public float Dx { get; }
            public float Dy { get; }
            public float DistanceSquared { get; }

            public ClusterFeature(float dx, float dy, float distanceSquared)
            {
                Dx = dx;
                Dy = dy;
                DistanceSquared = distanceSquared;
            }
        }
    }
}