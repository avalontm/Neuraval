using System.IO;
using System.Linq;
using Neuraval.Evolution.MarioBridge;
using Neuraval.Evolution.Neat;
using Neuraval.Evolution.Serialization;
using Xunit;

namespace Neuraval.Tests
{
    public class NavmContainerTests
    {
        [Fact]
        public void SaveLoad_RoundTrip_PreservesAllData()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), $"navm_test_{Guid.NewGuid():N}.navm");

            try
            {
                var headerJson = "{\"Kind\":\"test\",\"Version\":42}";
                var body = new byte[] { 0x01, 0x02, 0x03, 0xFF, 0xFE, 0xFD, 0x00 };

                NavmBinarySerializer.Save(tempPath, headerJson, body, compress: true);
                Assert.True(File.Exists(tempPath));

                var content = NavmBinarySerializer.Load(tempPath);
                Assert.NotNull(content);
                Assert.Equal(headerJson, content.HeaderJson);
                Assert.Equal(body, content.Body);
            }
            finally
            {
                if (File.Exists(tempPath)) File.Delete(tempPath);
            }
        }

        [Fact]
        public void SaveLoad_NoCompression_RoundTrip()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), $"navm_test_{Guid.NewGuid():N}.navm");

            try
            {
                var headerJson = "{\"Kind\":\"uncompressed\"}";
                var body = new byte[1024];
                Random.Shared.NextBytes(body);

                NavmBinarySerializer.Save(tempPath, headerJson, body, compress: false);
                var content = NavmBinarySerializer.Load(tempPath);

                Assert.NotNull(content);
                Assert.Equal(headerJson, content.HeaderJson);
                Assert.Equal(body, content.Body);
            }
            finally
            {
                if (File.Exists(tempPath)) File.Delete(tempPath);
            }
        }

        [Fact]
        public void Load_CorruptedChecksum_ReturnsNull()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), $"navm_test_{Guid.NewGuid():N}.navm");

            try
            {
                NavmBinarySerializer.Save(tempPath, "{}", new byte[] { 1, 2, 3 }, compress: false);

                var bytes = File.ReadAllBytes(tempPath);
                bytes[bytes.Length - 1] ^= 0xFF;
                File.WriteAllBytes(tempPath, bytes);

                var content = NavmBinarySerializer.Load(tempPath);
                Assert.Null(content);
            }
            finally
            {
                if (File.Exists(tempPath)) File.Delete(tempPath);
            }
        }

        [Fact]
        public void Load_WrongMagic_ReturnsNull()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), $"navm_test_{Guid.NewGuid():N}.navm");

            try
            {
                File.WriteAllBytes(tempPath, new byte[] { 0x00, 0x00, 0x00, 0x00, 0x01, 0x00 });
                var content = NavmBinarySerializer.Load(tempPath);
                Assert.Null(content);
            }
            finally
            {
                if (File.Exists(tempPath)) File.Delete(tempPath);
            }
        }

        [Fact]
        public void Load_NonexistentFile_ReturnsNull()
        {
            var content = NavmBinarySerializer.Load("D:/nonexistent/path/navm_test.navm");
            Assert.Null(content);
        }

        [Fact]
        public void LargePayload_RoundTrip()
        {
            var tempPath = Path.Combine(Path.GetTempPath(), $"navm_test_{Guid.NewGuid():N}.navm");

            try
            {
                var headerJson = "{\"Kind\":\"large\"}";
                var body = new byte[100_000];
                for (var i = 0; i < body.Length; i++)
                {
                    body[i] = (byte)(0x55 + (i % 32));
                }

                NavmBinarySerializer.Save(tempPath, headerJson, body, compress: true);
                var content = NavmBinarySerializer.Load(tempPath);

                Assert.NotNull(content);
                Assert.Equal(headerJson, content.HeaderJson);
                Assert.Equal(body, content.Body);

                var compressedSize = new FileInfo(tempPath).Length;
                Assert.True(compressedSize < body.Length, $"Compressed size ({compressedSize}) should be less than uncompressed ({body.Length})");
            }
            finally
            {
                if (File.Exists(tempPath)) File.Delete(tempPath);
            }
        }
    }

    public class MarioGenomeSerializerTests
    {
        [Fact]
        public void WriteRead_RoundTrip()
        {
            var tracker = new NeatInnovationTracker(0);
            var genome = NeatGenome.CreateInitial(4, 2, new Random(1234), tracker);

            using var stream = new MemoryStream();
            using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                MarioGenomeSerializer.Write(writer, genome);
            }

            stream.Position = 0;
            using var reader = new BinaryReader(stream);
            var loaded = MarioGenomeSerializer.Read(reader);

            Assert.Equal(genome.InputCount, loaded.InputCount);
            Assert.Equal(genome.OutputCount, loaded.OutputCount);
            Assert.Equal(genome.Nodes.Count, loaded.Nodes.Count);
            Assert.Equal(genome.Connections.Count, loaded.Connections.Count);

            for (var i = 0; i < genome.Nodes.Count; i++)
            {
                Assert.Equal(genome.Nodes[i].Id, loaded.Nodes[i].Id);
                Assert.Equal(genome.Nodes[i].Type, loaded.Nodes[i].Type);
            }

            for (var i = 0; i < genome.Connections.Count; i++)
            {
                Assert.Equal(genome.Connections[i].InNode, loaded.Connections[i].InNode);
                Assert.Equal(genome.Connections[i].OutNode, loaded.Connections[i].OutNode);
                Assert.Equal(genome.Connections[i].Weight, loaded.Connections[i].Weight);
                Assert.Equal(genome.Connections[i].Enabled, loaded.Connections[i].Enabled);
                Assert.Equal(genome.Connections[i].Innovation, loaded.Connections[i].Innovation);
            }
        }
    }

    public class MarioStateEncoderTests
    {
        [Fact]
        public void InputCount_MatchesExpected()
        {
            Assert.Equal(1324, MarioStateEncoder.InputCount);
        }

        [Fact]
        public void Layout_BlocksAreContiguous()
        {
            Assert.Equal(MarioStateEncoder.GridCellCount, MarioStateEncoder.MarioBlockStart);
            Assert.Equal(MarioStateEncoder.MarioBlockStart + MarioStateEncoder.MarioBlockSignalCount, MarioStateEncoder.CameraBlockStart);
            Assert.Equal(MarioStateEncoder.CameraBlockStart + MarioStateEncoder.CameraSignalCount, MarioStateEncoder.GameBlockStart);
            Assert.Equal(MarioStateEncoder.GameBlockStart + MarioStateEncoder.GameSignalCount, MarioStateEncoder.PowerupBlockStart);
            Assert.Equal(MarioStateEncoder.PowerupBlockStart + MarioStateEncoder.PowerupSignalCount, MarioStateEncoder.SpriteBlockStart);
            Assert.Equal(MarioStateEncoder.SpriteBlockStart + MarioStateEncoder.SpriteBlockSignalCount, MarioStateEncoder.ClusterBlockStart);
            Assert.Equal(MarioStateEncoder.ClusterBlockStart + MarioStateEncoder.ClusterBlockSignalCount, MarioStateEncoder.SecondPlayerBlockStart);
            Assert.Equal(MarioStateEncoder.SecondPlayerBlockStart + MarioStateEncoder.SecondPlayerBlockSignalCount, MarioStateEncoder.CoinBlockStart);
            Assert.Equal(MarioStateEncoder.CoinBlockStart + MarioStateEncoder.CoinSignalCount, MarioStateEncoder.DialogBlockStart);
            Assert.Equal(MarioStateEncoder.DialogBlockStart + MarioStateEncoder.DialogSignalCount, MarioStateEncoder.CliffBlockStart);
            Assert.Equal(MarioStateEncoder.CliffBlockStart + MarioStateEncoder.CliffSignalCount, MarioStateEncoder.YoshiCoinBlockStart);
            Assert.Equal(MarioStateEncoder.YoshiCoinBlockStart + MarioStateEncoder.YoshiCoinSignalCount, MarioStateEncoder.CarryBlockStart);
            Assert.Equal(MarioStateEncoder.CarryBlockStart + MarioStateEncoder.CarrySignalCount, MarioStateEncoder.CheckpointBlockStart);
            Assert.Equal(MarioStateEncoder.CheckpointBlockStart + MarioStateEncoder.CheckpointSignalCount, MarioStateEncoder.WallBlockStart);
            Assert.Equal(MarioStateEncoder.WallBlockStart + MarioStateEncoder.WallSignalCount, MarioStateEncoder.VerticalLevelBlockStart);
            Assert.Equal(MarioStateEncoder.VerticalLevelBlockStart + MarioStateEncoder.VerticalLevelSignalCount, MarioStateEncoder.PipeBlockStart);
            Assert.Equal(MarioStateEncoder.PipeBlockStart + MarioStateEncoder.PipeSignalCount, MarioStateEncoder.LevelBlockStart);
            Assert.Equal(MarioStateEncoder.LevelBlockStart + MarioStateEncoder.LevelSignalCount, MarioStateEncoder.HazardBlockStart);
            Assert.Equal(MarioStateEncoder.HazardBlockStart + MarioStateEncoder.HazardSignalCount, MarioStateEncoder.TileCategoryBlockStart);
            Assert.Equal(MarioStateEncoder.TileCategoryBlockStart + MarioStateEncoder.TileCategoryBlockSignalCount, MarioStateEncoder.InputCount);
        }

        [Fact]
        public void MarioBlockSignalCount_IsCorrect()
        {
            var expected = MarioStateEncoder.VelocitySignalCount
                + MarioStateEncoder.GroundedSignalCount
                + MarioStateEncoder.MarioStateSignalCount
                + MarioStateEncoder.MarioMiscSignalCount;

            Assert.Equal(MarioStateEncoder.MarioBlockSignalCount, expected);
            Assert.Equal(22, MarioStateEncoder.MarioBlockSignalCount);
        }

        [Fact]
        public void Encode_BlockedBitmask_DecodesToFourSignals()
        {
            var tiles = new byte[MarioStateEncoder.GridCellCount];
            var sprites = new SnesSprite[]
            {
                new(101, 200, 5, 0, 0, 0, blocked: 0x02 | 0x08),
            };
            var state = new SnesState(
                frame: 1, marioX: 100, marioY: 200, marioVelocityX: 0, marioVelocityY: 0,
                isDead: false, lives: 5, isLevelComplete: false, manualResetRequested: false,
                isGrounded: true, powerupLevel: 0, levelIndex: 0,
                tiles: tiles, sprites: sprites,
                direction: 0, blocked: 0x01 | 0x04, subPixelX: 0, subPixelY: 0,
                airState: 0, ducking: 0, climbing: 0, water: 0,
                pMeter: 0, takeoffMeter: 0, hurtTimer: 0, capeTimer: 0,
                cameraX: 0, cameraY: 0, gameMode: 0, levelMode: 0, translevel: 0,
                coins: 0, itemBox: 0, controller1: 0, controller1Prev: 0, controller2: 0, controller2Prev: 0);

            var input = MarioStateEncoder.Encode(state);

            var m = MarioStateEncoder.MarioBlockStart;
            Assert.Equal(0f, input[m + 4]);
            Assert.Equal(1f, input[m + 5]);
            Assert.Equal(0f, input[m + 6]);
            Assert.Equal(1f, input[m + 7]);

            var s0 = MarioStateEncoder.SpriteBlockStart;
            Assert.Equal(1f, input[s0 + 6]);
            Assert.Equal(0f, input[s0 + 7]);
            Assert.Equal(1f, input[s0 + 8]);
            Assert.Equal(0f, input[s0 + 9]);
        }

        [Fact]
        public void Encode_WallAndVerticalGaps_SurfaceOnDedicatedBlock()
        {
            var tiles = new byte[MarioStateEncoder.GridCellCount];
            var state = new SnesState(
                frame: 1, marioX: 100, marioY: 200, marioVelocityX: 0, marioVelocityY: 0,
                isDead: false, lives: 5, isLevelComplete: false, manualResetRequested: false,
                isGrounded: true, powerupLevel: 0, levelIndex: 0,
                tiles: tiles, sprites: Array.Empty<SnesSprite>(),
                direction: 0, blocked: 0, subPixelX: 0, subPixelY: 0,
                airState: 0, ducking: 0, climbing: 0, water: 0,
                pMeter: 0, takeoffMeter: 0, hurtTimer: 0, capeTimer: 0,
                cameraX: 0, cameraY: 0, gameMode: 0, levelMode: 0, translevel: 0,
                coins: 0, itemBox: 0, controller1: 0, controller1Prev: 0, controller2: 0, controller2Prev: 0,
                wallAheadDistance: 3, solidAboveDistance: 1, solidBelowDistance: 2);

            var input = MarioStateEncoder.Encode(state);

            var w = MarioStateEncoder.WallBlockStart;
            Assert.Equal(3f / MarioStateEncoder.WallDistanceScale, input[w + 0], 6);
            Assert.Equal(1f / MarioStateEncoder.WallDistanceScale, input[w + 1], 6);
            Assert.Equal(2f / MarioStateEncoder.WallDistanceScale, input[w + 2], 6);
            Assert.Equal(MarioStateEncoder.WallBlockStart + MarioStateEncoder.WallSignalCount, MarioStateEncoder.VerticalLevelBlockStart);
        }

        [Fact]
        public void Encode_VerticalLevelFlag_SurfacesOnDedicatedBlock()
        {
            var tiles = new byte[MarioStateEncoder.GridCellCount];
            var state = new SnesState(
                frame: 1, marioX: 100, marioY: 200, marioVelocityX: 0, marioVelocityY: 0,
                isDead: false, lives: 5, isLevelComplete: false, manualResetRequested: false,
                isGrounded: true, powerupLevel: 0, levelIndex: 0,
                tiles: tiles, sprites: Array.Empty<SnesSprite>(),
                direction: 0, blocked: 0, subPixelX: 0, subPixelY: 0,
                airState: 0, ducking: 0, climbing: 0, water: 0,
                pMeter: 0, takeoffMeter: 0, hurtTimer: 0, capeTimer: 0,
                cameraX: 0, cameraY: 0, gameMode: 0, levelMode: 0, translevel: 0,
                coins: 0, itemBox: 0, controller1: 0, controller1Prev: 0, controller2: 0, controller2Prev: 0,
                isVerticalLevel: true);

            var input = MarioStateEncoder.Encode(state);

            var vl = MarioStateEncoder.VerticalLevelBlockStart;
            Assert.Equal(1f, input[vl + 0]);
            Assert.Equal(MarioStateEncoder.InputCount, input.Length);
        }

        [Fact]
        public void Encode_PipeNear_SurfacesOnDedicatedBlock()
        {
            var tiles = new byte[MarioStateEncoder.GridCellCount];
            var state = new SnesState(
                frame: 1, marioX: 100, marioY: 200, marioVelocityX: 0, marioVelocityY: 0,
                isDead: false, lives: 5, isLevelComplete: false, manualResetRequested: false,
                isGrounded: true, powerupLevel: 0, levelIndex: 0,
                tiles: tiles, sprites: Array.Empty<SnesSprite>(),
                direction: 0, blocked: 0, subPixelX: 0, subPixelY: 0,
                airState: 0, ducking: 0, climbing: 0, water: 0,
                pMeter: 0, takeoffMeter: 0, hurtTimer: 0, capeTimer: 0,
                cameraX: 0, cameraY: 0, gameMode: 0, levelMode: 0, translevel: 0,
                coins: 0, itemBox: 0, controller1: 0, controller1Prev: 0, controller2: 0, controller2Prev: 0,
                pipeNear: 1, nearestPipeDx: 32, nearestPipeDy: -16);

            var input = MarioStateEncoder.Encode(state);

            var pp = MarioStateEncoder.PipeBlockStart;
            Assert.Equal(1f / MarioStateEncoder.CoinCountScale, input[pp + 0], 6);
            Assert.Equal(32f / MarioStateEncoder.SpriteOffsetScale, input[pp + 1], 6);
            Assert.Equal(-16f / MarioStateEncoder.SpriteOffsetScale, input[pp + 2], 6);
            Assert.Equal(MarioStateEncoder.InputCount, input.Length);
        }

        [Fact]
        public void Encode_NullTiles_ReturnsZeros()
        {
            var tiles = new byte[MarioStateEncoder.GridCellCount];
            var sprites = new SnesSprite[0];
            var state = new SnesState(
                frame: 1, marioX: 100, marioY: 200, marioVelocityX: 2, marioVelocityY: -3,
                isDead: false, lives: 5, isLevelComplete: false, manualResetRequested: false,
                isGrounded: true, powerupLevel: 1, levelIndex: 0,
                tiles: tiles, sprites: sprites,
                direction: 1, blocked: 0, subPixelX: 50, subPixelY: 30,
                airState: 0, ducking: 0, climbing: 0, water: 0,
                pMeter: 40, takeoffMeter: 0, hurtTimer: 0, capeTimer: 0,
                cameraX: 50, cameraY: 0, gameMode: 17, levelMode: 0, translevel: 0,
                coins: 25, itemBox: 0, controller1: 0x01, controller1Prev: 0x01, controller2: 0, controller2Prev: 0,
                marioSubSpeed: 60, reservedItemBox: 7);

            var input = MarioStateEncoder.Encode(state);
            Assert.Equal(MarioStateEncoder.InputCount, input.Length);

            Assert.Equal(2f / MarioStateEncoder.VelocityScale, input[MarioStateEncoder.MarioBlockStart + 0], 6);
            Assert.Equal(-3f / MarioStateEncoder.VelocityScale, input[MarioStateEncoder.MarioBlockStart + 1], 6);
            Assert.Equal(1f, input[MarioStateEncoder.MarioBlockStart + 2], 6);
            Assert.Equal(1, input[MarioStateEncoder.MarioBlockStart + 3]);
            Assert.Equal(50f / MarioStateEncoder.SubpixelScale, input[MarioStateEncoder.MarioBlockStart + 8], 6);
            Assert.Equal(60f / MarioStateEncoder.SubpixelScale, input[MarioStateEncoder.MarioBlockStart + 10], 6);
            Assert.Equal(40f / MarioStateEncoder.PMeterScale, input[MarioStateEncoder.MarioBlockStart + 15], 6);
            Assert.Equal(25f / MarioStateEncoder.CoinsScale, input[MarioStateEncoder.MarioBlockStart + 19], 6);
            Assert.Equal(7f / MarioStateEncoder.CoinsScale, input[MarioStateEncoder.MarioBlockStart + 21], 6);

            var screenX = (100 - 50) / MarioStateEncoder.ScreenXScale;
            Assert.Equal(screenX, input[MarioStateEncoder.CameraBlockStart + 0], 6);
            Assert.Equal((100f - 0) / MarioStateEncoder.AbsoluteXScale, input[MarioStateEncoder.CameraBlockStart + 2], 6);

            Assert.Equal(17f / 255f, input[MarioStateEncoder.GameBlockStart + 0], 6);

            Assert.Equal(1f, input[MarioStateEncoder.PowerupBlockStart + 1], 6);
            Assert.Equal(0f, input[MarioStateEncoder.PowerupBlockStart + 0], 6);

            Assert.Equal(0f, input[MarioStateEncoder.SpriteBlockStart + 0], 6);
        }

        [Fact]
        public void NearestSprites_ReturnsByDistance()
        {
            var tiles = new byte[MarioStateEncoder.GridCellCount];
            var sprites = new SnesSprite[]
            {
                new(200, 200, 1, 0, 0, 0, 0),
                new(105, 200, 2, 0, 0, 0, 0, subPixelX: 128),
                new(103, 201, 3, 0, 0, 0, 0, subPixelX: 64, subPixelY: 32),
            };
            var state = new SnesState(
                frame: 1, marioX: 100, marioY: 200, marioVelocityX: 0, marioVelocityY: 0,
                isDead: false, lives: 5, isLevelComplete: false, manualResetRequested: false,
                isGrounded: true, powerupLevel: 0, levelIndex: 0,
                tiles: tiles, sprites: sprites,
                direction: 0, blocked: 0, subPixelX: 0, subPixelY: 0,
                airState: 0, ducking: 0, climbing: 0, water: 0,
                pMeter: 0, takeoffMeter: 0, hurtTimer: 0, capeTimer: 0,
                cameraX: 0, cameraY: 0, gameMode: 0, levelMode: 0, translevel: 0,
                coins: 0, itemBox: 0, controller1: 0, controller1Prev: 0, controller2: 0, controller2Prev: 0);

            var input = MarioStateEncoder.Encode(state);

            var s0 = MarioStateEncoder.SpriteBlockStart;
            var stride = MarioStateEncoder.SignalsPerSprite;
            Assert.Equal(3f / MarioStateEncoder.SpriteOffsetScale, input[s0 + 0], 6);
            Assert.Equal(1f / MarioStateEncoder.SpriteOffsetScale, input[s0 + 1], 6);
            Assert.Equal(3f / MarioStateEncoder.SpriteTypeScale, input[s0 + 2], 6);
            Assert.Equal(64f / MarioStateEncoder.SubpixelScale, input[s0 + 10], 6);
            Assert.Equal(32f / MarioStateEncoder.SubpixelScale, input[s0 + 11], 6);

            Assert.Equal(5f / MarioStateEncoder.SpriteOffsetScale, input[s0 + stride + 0], 6);
            Assert.Equal(0f / MarioStateEncoder.SpriteOffsetScale, input[s0 + stride + 1], 6);
            Assert.Equal(2f / MarioStateEncoder.SpriteTypeScale, input[s0 + stride + 2], 6);
            Assert.Equal(128f / MarioStateEncoder.SubpixelScale, input[s0 + stride + 10], 6);

            Assert.Equal(100f / MarioStateEncoder.SpriteOffsetScale, input[s0 + stride * 2 + 0], 6);
            Assert.Equal(0f / MarioStateEncoder.SpriteOffsetScale, input[s0 + stride * 2 + 1], 6);
            Assert.Equal(1f / MarioStateEncoder.SpriteTypeScale, input[s0 + stride * 2 + 2], 6);
        }

        [Fact]
        public void Encode_StatusStunFlagsMisc_SurfaceOnSpriteBlock()
        {
            var tiles = new byte[MarioStateEncoder.GridCellCount];
            var sprites = new SnesSprite[]
            {
                new(101, 200, 5, 0, 0, 0, 0, 0, 0, 0, status: 8, stunTimer: 40, properties: 0x36, misc1: 11, misc2: 22, misc3: 33, offscreenFull: 44, eaten: 55, objectInteraction: 66, spinTimer: 77),
            };
            var state = new SnesState(
                frame: 1, marioX: 100, marioY: 200, marioVelocityX: 0, marioVelocityY: 0,
                isDead: false, lives: 5, isLevelComplete: false, manualResetRequested: false,
                isGrounded: true, powerupLevel: 0, levelIndex: 0,
                tiles: tiles, sprites: sprites,
                direction: 0, blocked: 0, subPixelX: 0, subPixelY: 0,
                airState: 0, ducking: 0, climbing: 0, water: 0,
                pMeter: 0, takeoffMeter: 0, hurtTimer: 0, capeTimer: 0,
                cameraX: 0, cameraY: 0, gameMode: 0, levelMode: 0, translevel: 0,
                coins: 0, itemBox: 0, controller1: 0, controller1Prev: 0, controller2: 0, controller2Prev: 0);

            var input = MarioStateEncoder.Encode(state);

            var s0 = MarioStateEncoder.SpriteBlockStart;
            Assert.Equal(8f / MarioStateEncoder.StatusScale, input[s0 + 12], 6);
            Assert.Equal(40f / MarioStateEncoder.MeterScale, input[s0 + 13], 6);
            Assert.Equal(1f, input[s0 + 14], 6);
            Assert.Equal(1f, input[s0 + 15], 6);
            Assert.Equal(1f, input[s0 + 16], 6);
            Assert.Equal(1f, input[s0 + 17], 6);
            Assert.Equal(11f / MarioStateEncoder.MeterScale, input[s0 + 18], 6);
            Assert.Equal(22f / MarioStateEncoder.MeterScale, input[s0 + 19], 6);
            Assert.Equal(33f / MarioStateEncoder.MeterScale, input[s0 + 20], 6);
            Assert.Equal(44f / MarioStateEncoder.MeterScale, input[s0 + 21], 6);
            Assert.Equal(55f / MarioStateEncoder.MeterScale, input[s0 + 22], 6);
            Assert.Equal(66f / MarioStateEncoder.MeterScale, input[s0 + 23], 6);
            Assert.Equal(77f / MarioStateEncoder.MeterScale, input[s0 + 24], 6);
            Assert.Equal(1f / MarioStateEncoder.SpriteOffsetScale, input[s0 + 0], 6);
        }

        [Fact]
        public void Encode_GameBlock_SurfacesNewSignals()
        {
            var tiles = new byte[MarioStateEncoder.GridCellCount];
            var state = new SnesState(
                frame: 1, marioX: 100, marioY: 200, marioVelocityX: 0, marioVelocityY: 0,
                isDead: false, lives: 5, isLevelComplete: false, manualResetRequested: false,
                isGrounded: true, powerupLevel: 0, levelIndex: 0,
                tiles: tiles, sprites: Array.Empty<SnesSprite>(),
                direction: 0, blocked: 0, subPixelX: 0, subPixelY: 0,
                airState: 0, ducking: 0, climbing: 0, water: 0,
                pMeter: 0, takeoffMeter: 0, hurtTimer: 0, capeTimer: 0,
                cameraX: 0, cameraY: 0, gameMode: 17, levelMode: 0, translevel: 0,
                coins: 0, itemBox: 0, controller1: 0, controller1Prev: 0, controller2: 0, controller2Prev: 0,
                bluePowTimer: 120, silverPowTimer: 60, doorExitCounter: 3,
                itemMemory: 30, currentPlayer: 1, character: 1, currentPlayerCoins: 45);

            var input = MarioStateEncoder.Encode(state);

            var g = MarioStateEncoder.GameBlockStart;
            Assert.Equal(17f / 255f, input[g + 0], 6);
            Assert.Equal(120f / MarioStateEncoder.MeterScale, input[g + 3], 6);
            Assert.Equal(60f / MarioStateEncoder.MeterScale, input[g + 4], 6);
            Assert.Equal(3f / MarioStateEncoder.MeterScale, input[g + 5], 6);
            Assert.Equal(30f / MarioStateEncoder.MeterScale, input[g + 6], 6);
            Assert.Equal(1f / MarioStateEncoder.MeterScale, input[g + 7], 6);
            Assert.Equal(1f / MarioStateEncoder.MeterScale, input[g + 8], 6);
            Assert.Equal(45f / MarioStateEncoder.MeterScale, input[g + 9], 6);
        }

        [Fact]
        public void Encode_NearestClustersAndLayers()
        {
            var tiles = new byte[MarioStateEncoder.GridCellCount];
            var state = new SnesState(
                frame: 1, marioX: 100, marioY: 200, marioVelocityX: 0, marioVelocityY: 0,
                isDead: false, lives: 5, isLevelComplete: false, manualResetRequested: false,
                isGrounded: true, powerupLevel: 0, levelIndex: 0,
                tiles: tiles, sprites: Array.Empty<SnesSprite>(),
                direction: 0, blocked: 0, subPixelX: 0, subPixelY: 0,
                airState: 0, ducking: 0, climbing: 0, water: 0,
                pMeter: 0, takeoffMeter: 0, hurtTimer: 0, capeTimer: 0,
                cameraX: 50, cameraY: 0, gameMode: 0, levelMode: 0, translevel: 0,
                coins: 0, itemBox: 0, controller1: 0, controller1Prev: 0, controller2: 0, controller2Prev: 0,
                clusterSprites: new[]
                {
                    new SnesSprite(500, 300, 0),
                    new SnesSprite(120, 205, 0),
                    new SnesSprite(700, 260, 0),
                },
                layer2X: 60, layer2Y: 4, layer3X: 70, layer3Y: 8,
                p2Controller1: 0x81, p2Controller1Prev: 0x41, p2Controller2: 0x80, p2Controller2Prev: 0x40);

            var input = MarioStateEncoder.Encode(state);

            var t = MarioStateEncoder.SecondPlayerBlockStart;
            Assert.Equal(129f / MarioStateEncoder.StatusScale, input[t + 0], 6);
            Assert.Equal(65f / MarioStateEncoder.StatusScale, input[t + 1], 6);
            Assert.Equal(128f / MarioStateEncoder.StatusScale, input[t + 2], 6);
            Assert.Equal(64f / MarioStateEncoder.StatusScale, input[t + 3], 6);

            var c = MarioStateEncoder.CameraBlockStart;
            Assert.Equal((60 - 50f) / MarioStateEncoder.ScreenXScale, input[c + 3], 6);
            Assert.Equal(4f / MarioStateEncoder.ScreenYScale, input[c + 4], 6);
            Assert.Equal((70 - 50f) / MarioStateEncoder.ScreenXScale, input[c + 5], 6);
            Assert.Equal(8f / MarioStateEncoder.ScreenYScale, input[c + 6], 6);

            var k = MarioStateEncoder.ClusterBlockStart;
            Assert.Equal(20f / MarioStateEncoder.SpriteOffsetScale, input[k + 0], 6);
            Assert.Equal(5f / MarioStateEncoder.SpriteOffsetScale, input[k + 1], 6);
            Assert.Equal(400f / MarioStateEncoder.SpriteOffsetScale, input[k + 2], 6);
            Assert.Equal(100f / MarioStateEncoder.SpriteOffsetScale, input[k + 3], 6);
            Assert.Equal(600f / MarioStateEncoder.SpriteOffsetScale, input[k + 4], 6);
            Assert.Equal(60f / MarioStateEncoder.SpriteOffsetScale, input[k + 5], 6);
        }

        [Fact]
        public void Encode_CoinSignalsAndRawTiles_SurfaceOnCoinBlock()
        {
            var tiles = new byte[MarioStateEncoder.GridCellCount];
            tiles[85] = 0x7F;
            var state = new SnesState(
                frame: 1, marioX: 100, marioY: 200, marioVelocityX: 0, marioVelocityY: 0,
                isDead: false, lives: 5, isLevelComplete: false, manualResetRequested: false,
                isGrounded: true, powerupLevel: 0, levelIndex: 0,
                tiles: tiles, sprites: Array.Empty<SnesSprite>(),
                direction: 0, blocked: 0, subPixelX: 0, subPixelY: 0,
                airState: 0, ducking: 0, climbing: 0, water: 0,
                pMeter: 0, takeoffMeter: 0, hurtTimer: 0, capeTimer: 0,
                cameraX: 0, cameraY: 0, gameMode: 0, levelMode: 0, translevel: 0,
                coins: 0, itemBox: 0, controller1: 0, controller1Prev: 0, controller2: 0, controller2Prev: 0,
                coinsNear: 4, nearestCoinDx: 24, nearestCoinDy: -48,
                coinBlocksNear: 2, nearestCoinBlockDx: 0, nearestCoinBlockDy: -16,
                dialogNear: 1, nearestDialogDx: -8, nearestDialogDy: -48,
                cliffGaps: new[] { new CliffGap(2, 3), new CliffGap(5, 2) });

            var input = MarioStateEncoder.Encode(state);

            Assert.Equal(0f, input[0], 6);
            Assert.Equal(0x7F / 255f, input[85], 6);

            var cn = MarioStateEncoder.CoinBlockStart;
            Assert.Equal(4f / MarioStateEncoder.CoinCountScale, input[cn + 0], 6);
            Assert.Equal(24f / MarioStateEncoder.CoinOffsetScale, input[cn + 1], 6);
            Assert.Equal(-48f / MarioStateEncoder.CoinOffsetScale, input[cn + 2], 6);
            Assert.Equal(2f / MarioStateEncoder.CoinCountScale, input[cn + 3], 6);
            Assert.Equal(0f, input[cn + 4], 6);
            Assert.Equal(-16f / MarioStateEncoder.CoinOffsetScale, input[cn + 5], 6);

            var dl = MarioStateEncoder.DialogBlockStart;
            Assert.Equal(1f / MarioStateEncoder.CoinCountScale, input[dl + 0], 6);
            Assert.Equal(-8f / MarioStateEncoder.CoinOffsetScale, input[dl + 1], 6);
            Assert.Equal(-48f / MarioStateEncoder.CoinOffsetScale, input[dl + 2], 6);

            var cl = MarioStateEncoder.CliffBlockStart;
            Assert.Equal(2f / 7f, input[cl + 0], 6);
            Assert.Equal(3f / 8f, input[cl + 1], 6);
            Assert.Equal(5f / 7f, input[cl + 2], 6);
            Assert.Equal(2f / 8f, input[cl + 3], 6);
        }

        [Fact]
        public void Encode_YoshiCoinCarryAndCheckpoint_SurfaceOnDedicatedBlocks()
        {
            var tiles = new byte[MarioStateEncoder.GridCellCount];
            var sprites = new SnesSprite[]
            {
                new(120, 190, MarioSpriteNames.YoshiCoinSpriteId),
                new(90, 205, 0x74, status: MarioSpriteStatus.StationaryCarryable),
            };
            var state = new SnesState(
                frame: 1, marioX: 100, marioY: 200, marioVelocityX: 0, marioVelocityY: 0,
                isDead: false, lives: 5, isLevelComplete: false, manualResetRequested: false,
                isGrounded: true, powerupLevel: 0, levelIndex: 0,
                tiles: tiles, sprites: sprites,
                direction: 0, blocked: 0, subPixelX: 0, subPixelY: 0,
                airState: 0, ducking: 0, climbing: 0, water: 0,
                pMeter: 0, takeoffMeter: 0, hurtTimer: 0, capeTimer: 0,
                cameraX: 0, cameraY: 0, gameMode: 0, levelMode: 0, translevel: 0,
                coins: 0, itemBox: 0, controller1: 0, controller1Prev: 0, controller2: 0, controller2Prev: 0,
                carryingFlag: 1, holdingObjectFlag: 1, midwayPointFlag: 1);

            var input = MarioStateEncoder.Encode(state);

            var yc = MarioStateEncoder.YoshiCoinBlockStart;
            Assert.Equal(1f / MarioStateEncoder.CoinCountScale, input[yc + 0], 6);
            Assert.Equal(20f / MarioStateEncoder.SpriteOffsetScale, input[yc + 1], 6);
            Assert.Equal(-10f / MarioStateEncoder.SpriteOffsetScale, input[yc + 2], 6);

            var ca = MarioStateEncoder.CarryBlockStart;
            Assert.Equal(1f, input[ca + 0], 6);
            Assert.Equal(1f, input[ca + 1], 6);
            Assert.Equal(1f, input[ca + 2], 6);
            Assert.Equal(-10f / MarioStateEncoder.SpriteOffsetScale, input[ca + 3], 6);
            Assert.Equal(5f / MarioStateEncoder.SpriteOffsetScale, input[ca + 4], 6);

            var chk = MarioStateEncoder.CheckpointBlockStart;
            Assert.Equal(1f, input[chk + 0], 6);
        }
    }

    public class SnesStateParseTests
    {
        [Fact]
        public void Parse_FullWireLine_70Fields()
        {
            var tileValues = Enumerable.Repeat(0, SnesState.GridSize * SnesState.GridSize).ToList();
            tileValues[0] = 0x2B;
            var tiles = string.Join(",", tileValues);
            var wire = string.Join("|", new[]
            {
                "123", "100", "200", "16", "-8", "0", "4", tiles, "10,176,15,0,0,1,1,0,64,32,8,5,54,11,22,33,200,201,202,203",
                "500,300;610,220",
                "1", "0", "0", "1", "0",
                "1", "0", "50", "30", "0", "0", "0", "0", "40", "0", "0", "0",
                "15", "16", "17",
                "50", "0", "55", "45", "66", "56",
                "17", "0", "0",
                "6", "7", "8", "9",
                "25", "0",
                "5", "0", "2", "0",
                "60", "7", "5", "2", "9", "1", "456",
                "1", "2", "3", "4",
                "3;20;-40;2;1;80",
                "1;-8;-48",
                "2;3;1;2",
                "1", "0", "1", "3",
                "3;1;2",
                "1",
                "1;12;-6"
            });

            var state = SnesState.Parse(wire);

            Assert.Equal(123, state.Frame);
            Assert.Equal(100, state.MarioX);
            Assert.Equal(200, state.MarioY);
            Assert.Equal(16, state.MarioVelocityX);
            Assert.Equal(-8, state.MarioVelocityY);
            Assert.True(state.IsGrounded);
            Assert.Equal(4, state.Lives);
            Assert.Equal(SnesState.GridSize * SnesState.GridSize, state.Tiles.Count);
            Assert.Equal(0x2B, state.Tiles[0]);
            Assert.Single(state.Sprites);

            var sprite = state.Sprites[0];
            Assert.Equal(10, sprite.X);
            Assert.Equal(176, sprite.Y);
            Assert.Equal(15, sprite.Type);
            Assert.Equal(1, sprite.Direction);
            Assert.Equal(64, sprite.SubPixelX);
            Assert.Equal(32, sprite.SubPixelY);
            Assert.Equal(8, sprite.Status);
            Assert.Equal(5, sprite.StunTimer);
            Assert.Equal(0x36, sprite.Properties);
            Assert.Equal(11, sprite.Misc1);
            Assert.Equal(22, sprite.Misc2);
            Assert.Equal(33, sprite.Misc3);
            Assert.Equal(200, sprite.OffscreenFull);
            Assert.Equal(201, sprite.Eaten);
            Assert.Equal(202, sprite.ObjectInteraction);
            Assert.Equal(203, sprite.SpinTimer);

            Assert.Equal(2, state.ClusterSprites.Count);
            Assert.Equal(500, state.ClusterSprites[0].X);
            Assert.Equal(300, state.ClusterSprites[0].Y);

            Assert.Equal(15, state.BluePowTimer);
            Assert.Equal(16, state.SilverPowTimer);
            Assert.Equal(17, state.DoorExitCounter);

            Assert.Equal(55, state.Layer2X);
            Assert.Equal(45, state.Layer2Y);
            Assert.Equal(66, state.Layer3X);
            Assert.Equal(56, state.Layer3Y);

            Assert.Equal(6, state.CurrentPlayer);
            Assert.Equal(7, state.Character);
            Assert.Equal(8, state.ItemMemory);
            Assert.Equal(9, state.CurrentPlayerCoins);

            Assert.Equal(1, state.PowerupLevel);
            Assert.Equal(25, state.Coins);
            Assert.Equal(60, state.MarioSubSpeed);
            Assert.Equal(7, state.ReservedItemBox);
            Assert.Equal(5, state.Controller1Copy);
            Assert.Equal(2, state.Controller2Copy);
            Assert.Equal(9, state.SpriteFrame);
            Assert.Equal(1, state.Lag);
            Assert.Equal(456, state.EmuFrame);
            Assert.Equal(1, state.P2Controller1);
            Assert.Equal(2, state.P2Controller1Prev);
            Assert.Equal(3, state.P2Controller2);
            Assert.Equal(4, state.P2Controller2Prev);

            Assert.Equal(3, state.CoinsNear);
            Assert.Equal(20, state.NearestCoinDx);
            Assert.Equal(-40, state.NearestCoinDy);
            Assert.Equal(2, state.CoinBlocksNear);
            Assert.Equal(1, state.NearestCoinBlockDx);
            Assert.Equal(80, state.NearestCoinBlockDy);

            Assert.Equal(1, state.DialogNear);
            Assert.Equal(-8, state.NearestDialogDx);
            Assert.Equal(-48, state.NearestDialogDy);

            Assert.Equal(2, state.CliffGaps.Count);
            Assert.Equal(2, state.CliffGaps[0].StartTiles);
            Assert.Equal(3, state.CliffGaps[0].WidthTiles);
            Assert.Equal(1, state.CliffGaps[1].StartTiles);
            Assert.Equal(2, state.CliffGaps[1].WidthTiles);

            Assert.Equal(1, state.CarryingFlag);
            Assert.Equal(0, state.HoldingObjectFlag);
            Assert.True(state.IsHoldingItem);
            Assert.Equal(1, state.MidwayPointFlag);
            Assert.True(state.MidwayPointReached);
            Assert.Equal(3, state.YoshiCoinsCollected);

            Assert.Equal(3, state.WallAheadDistance);
            Assert.Equal(1, state.SolidAboveDistance);
            Assert.Equal(2, state.SolidBelowDistance);
            Assert.True(state.IsVerticalLevel);

            Assert.Equal(1, state.PipeNear);
            Assert.Equal(12, state.NearestPipeDx);
            Assert.Equal(-6, state.NearestPipeDy);
        }

        [Fact]
        public void Parse_WrongFieldCount_Throws()
        {
            Assert.Throws<FormatException>(() => SnesState.Parse("0|1|2"));
        }

        [Fact]
        public void Parse_EmptySprites_ReturnsEmptyList()
        {
            var tiles = string.Join(",", Enumerable.Repeat("0", SnesState.GridSize * SnesState.GridSize));
            var wire = string.Join("|", new[]
            {
                "1", "0", "0", "0", "0", "0", "4", tiles, "", "", "1", "0", "0", "0", "0",
                "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0",
                "0", "0", "0",
                "0", "0", "0", "0", "0", "0",
                "0", "0", "0",
                "0", "0", "0", "0",
                "0", "0",
                "0", "0", "0", "0",
                "0", "0", "0", "0", "0", "0", "0",
                "0", "0", "0", "0",
                "0;0;0;0;0;0",
                "0;0;0",
                "",
                "0", "0", "0", "0",
                "0;0;0",
                "0",
                "0;0;0"
            });

            var state = SnesState.Parse(wire);
            Assert.Empty(state.Sprites);
            Assert.Empty(state.ClusterSprites);
            Assert.Equal(0, state.CoinsNear);
            Assert.Equal(0, state.NearestCoinBlockDx);
            Assert.Empty(state.CliffGaps);
            Assert.False(state.IsHoldingItem);
            Assert.False(state.MidwayPointReached);
            Assert.Equal(0, state.YoshiCoinsCollected);
            Assert.Equal(0, state.WallAheadDistance);
            Assert.Equal(0, state.SolidAboveDistance);
            Assert.Equal(0, state.SolidBelowDistance);
            Assert.False(state.IsVerticalLevel);
            Assert.Equal(0, state.PipeNear);
        }
    }

    public class MarioAgentOutputTests
    {
        [Fact]
        public void OutputCount_IsSeven()
        {
            Assert.Equal(7, MarioAgentOutput.Count);
        }

        [Fact]
        public void ToAction_AllNegative_ReturnsNone()
        {
            var output = new float[] { -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f };
            var action = MarioAgentOutput.ToAction(output);
            Assert.Equal(SnesButton.None, action.Buttons);
        }

        [Fact]
        public void ToAction_DownPressed()
        {
            var output = new float[] { -0.1f, -0.1f, -0.1f, -0.1f, -0.1f, 0.9f };
            var action = MarioAgentOutput.ToAction(output);
            Assert.True(action.IsPressed(SnesButton.Down));
            Assert.False(action.IsPressed(SnesButton.B));
        }
    }

    public class MarioControllerEncoderTests
    {
        [Fact]
        public void Decode_LeftRight_A_B_Y_Down()
        {
            var ctrl1 = 0x02 | 0x04 | 0x40 | 0x80;
            var buttons = MarioControllerEncoder.Decode(ctrl1, 0);
            Assert.True(buttons.HasFlag(SnesButton.Left));
            Assert.True(buttons.HasFlag(SnesButton.Down));
            Assert.True(buttons.HasFlag(SnesButton.Y));
            Assert.True(buttons.HasFlag(SnesButton.B));
            Assert.False(buttons.HasFlag(SnesButton.Right));
        }

        [Fact]
        public void Decode_Controller2_A_X_L_R()
        {
            var ctrl2 = 0x10 | 0x20 | 0x40 | 0x80;
            var buttons = MarioControllerEncoder.Decode(0, ctrl2);
            Assert.True(buttons.HasFlag(SnesButton.R));
            Assert.True(buttons.HasFlag(SnesButton.L));
            Assert.True(buttons.HasFlag(SnesButton.X));
            Assert.True(buttons.HasFlag(SnesButton.A));
        }
    }
}