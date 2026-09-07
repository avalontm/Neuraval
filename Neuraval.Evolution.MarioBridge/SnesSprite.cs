namespace Neuraval.Evolution.MarioBridge
{
    // Valores del campo Status de sprite ($7E:14C8, tabla de estado de sprites)
    // relevantes para items sostenibles/tirables. Ver SMW_RAM_Map_IA.md.
    public static class MarioSpriteStatus
    {
        public const int StationaryCarryable = 0x09;
        public const int Kicked = 0x0A;
        public const int Carried = 0x0B;
    }

    public readonly struct SnesSprite
    {
        public int X { get; }
        public int Y { get; }
        public int Type { get; }
        public int VelocityX { get; }
        public int VelocityY { get; }
        public int Direction { get; }
        public int Blocked { get; }
        public int Offscreen { get; }
        public int SubPixelX { get; }
        public int SubPixelY { get; }
        public int Status { get; }
        public int StunTimer { get; }
        public int Properties { get; }
        public int Misc1 { get; }
        public int Misc2 { get; }
        public int Misc3 { get; }
        public int OffscreenFull { get; }
        public int Eaten { get; }
        public int ObjectInteraction { get; }
        public int SpinTimer { get; }

        public SnesSprite(
            int x,
            int y,
            int type,
            int velocityX = 0,
            int velocityY = 0,
            int direction = 0,
            int blocked = 0,
            int offscreen = 0,
            int subPixelX = 0,
            int subPixelY = 0,
            int status = 0,
            int stunTimer = 0,
            int properties = 0,
            int misc1 = 0,
            int misc2 = 0,
            int misc3 = 0,
            int offscreenFull = 0,
            int eaten = 0,
            int objectInteraction = 0,
            int spinTimer = 0)
        {
            X = x;
            Y = y;
            Type = type;
            VelocityX = velocityX;
            VelocityY = velocityY;
            Direction = direction;
            Blocked = blocked;
            Offscreen = offscreen;
            SubPixelX = subPixelX;
            SubPixelY = subPixelY;
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
        }
    }
}