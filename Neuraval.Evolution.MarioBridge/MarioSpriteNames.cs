using System.Collections.Generic;

namespace Neuraval.Evolution.MarioBridge
{
    public static class MarioSpriteNames
    {
        // Moneda Yoshi (Dragon Coin): inferido con alta confianza por posicion
        // en la tabla de sprites de SMW (encaja entre $C3 Porcu-Puffer y $C5
        // Boo Grande jefe, tal como en el mapa de referencia). Verificar en
        // BizHawk con el visor de sprites si se detecta algo raro en juego.
        public const int YoshiCoinSpriteId = 0xC4;

        private static readonly Dictionary<int, string> Known = new()
        {
            { 0x00, "Koopa verde sin caparazon" },
            { 0x01, "Koopa rojo sin caparazon" },
            { 0x02, "Koopa azul sin caparazon" },
            { 0x03, "Koopa amarillo sin caparazon" },
            { 0x04, "Koopa verde" },
            { 0x05, "Koopa rojo" },
            { 0x06, "Koopa azul" },
            { 0x07, "Koopa amarillo" },
            { 0x08, "Koopa verde volador" },
            { 0x09, "Koopa verde saltarin" },
            { 0x0A, "Koopa rojo volador vertical" },
            { 0x0B, "Koopa rojo volador horizontal" },
            { 0x0C, "Koopa amarillo alado" },
            { 0x0D, "Bob-omb" },
            { 0x0E, "Keyhole" },
            { 0x0F, "Goomba" },
            { 0x10, "Goomba alado" },
            { 0x11, "Buzzy Beetle" },
            { 0x13, "Spiny" },
            { 0x14, "Spiny cayendo" },
            { 0x15, "Cheep Cheep" },
            { 0x16, "Cheep Cheep vertical" },
            { 0x17, "Cheep Cheep volador" },
            { 0x18, "Cheep Cheep saltando" },
            { 0x1A, "Planta Piranha" },
            { 0x1B, "Balon rebotador" },
            { 0x1C, "Bala Bill" },
            { 0x1D, "Llama saltarina" },
            { 0x1E, "Lakitu" },
            { 0x1F, "Magikoopa" },
            { 0x22, "Koopa red vertical verde" },
            { 0x23, "Koopa red vertical rojo" },
            { 0x24, "Koopa red horizontal verde" },
            { 0x25, "Koopa red horizontal rojo" },
            { 0x26, "Thwomp" },
            { 0x27, "Thwimp" },
            { 0x28, "Boo Grande" },
            { 0x2A, "Planta Piranha invertida" },
            { 0x2E, "Spike Top" },
            { 0x30, "Dry Bones" },
            { 0x31, "Bony Beetle" },
            { 0x32, "Dry Bones de precipicio" },
            { 0x33, "Podoboo" },
            { 0x34, "Bola de fuego de jefe" },
            { 0x35, "Yoshi" },
            { 0x37, "Boo" },
            { 0x38, "Eerie" },
            { 0x39, "Eerie ondulante" },
            { 0x3A, "Urchin" },
            { 0x3B, "Urchin entre paredes" },
            { 0x3C, "Urchin que sigue paredes" },
            { 0x3D, "Rip Van Fish" },
            { 0x3F, "Para-Goomba" },
            { 0x40, "Para-Bomb" },
            { 0x41, "Delfin" },
            { 0x42, "Delfin vaivien" },
            { 0x43, "Delfin arriba/abajo" },
            { 0x44, "Torpedo Ted" },
            { 0x4D, "Monty Mole" },
            { 0x4E, "Monty Mole de risco" },
            { 0x4F, "Planta Piranha saltarina" },
            { 0x50, "Planta Piranha con fuego" },
            { 0x51, "Ninji" },
            { 0x6E, "Dino-Rhino" },
            { 0x6F, "Dino-Torch" },
            { 0x70, "Pokey" },
            { 0x71, "Super Koopa capa roja" },
            { 0x72, "Super Koopa capa amarilla" },
            { 0x73, "Super Koopa en tierra" },
            { 0x74, "Champinon" },
            { 0x75, "Flor de fuego" },
            { 0x76, "Estrella" },
            { 0x77, "Pluma capa" },
            { 0x78, "1-Up" },
            { 0x86, "Wiggler" },
            { 0x91, "Chargin' Chuck" },
            { 0x99, "Volcano Lotus" },
            { 0x9A, "Sumo Brother" },
            { 0x9B, "Hammer Brother" },
            { 0x9E, "Bola y cadena" },
            { 0x9F, "Banzai Bill" },
            { 0xA1, "Bola de Bowser" },
            { 0xA2, "Mecha-Koopa" },
            { 0xA6, "Hothead" },
            { 0xA8, "Blargg" },
            { 0xAB, "Rex" },
            { 0xAC, "Pua de madera hacia abajo" },
            { 0xAD, "Pua de madera hacia arriba" },
            { 0xB2, "Pua cayendo" },
            { 0xB4, "Amoladora" },
            { 0xBD, "Koopa deslizante" },
            { 0xBE, "Swooper" },
            { 0xBF, "Mega Mole" },
            { 0xC3, "Porcu-Puffer" },
            { YoshiCoinSpriteId, "Moneda Yoshi" },
            { 0xC5, "Boo Grande jefe" },
            { 0xDA, "Caparazon verde" },
            { 0xDB, "Caparazon rojo" },
            { 0xDC, "Caparazon azul" },
            { 0xDD, "Caparazon amarillo" },
            { 0xDF, "Para-caparazon verde" }
        };

        public static string Name(int type)
        {
            return Known.TryGetValue(type, out var name) ? name : $"sprite ${type:X2}";
        }

        public static bool IsYoshiCoin(int type)
        {
            return type == YoshiCoinSpriteId;
        }
    }
}