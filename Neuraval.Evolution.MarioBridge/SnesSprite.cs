namespace Neuraval.Evolution.MarioBridge
{
    public readonly struct SnesSprite
    {
        public int X { get; }
        public int Y { get; }

        // ID de tipo de sprite de SMW (direccion $7E:009E + slot). Ej: distintos
        // valores para Goomba, Koopa, hongo, etc. No lo clasificamos nosotros
        // como "enemigo" o "item" - se lo pasamos crudo a la red y que la
        // evolucion aprenda la asociacion (mas confiable que adivinar tablas).
        public int Type { get; }

        public SnesSprite(int x, int y, int type)
        {
            X = x;
            Y = y;
            Type = type;
        }
    }
}
