namespace Neuraval.Evolution.MarioBridge
{
    public readonly struct CliffGap
    {
        public int StartTiles { get; }
        public int WidthTiles { get; }

        public CliffGap(int startTiles, int widthTiles)
        {
            StartTiles = startTiles;
            WidthTiles = widthTiles;
        }
    }
}