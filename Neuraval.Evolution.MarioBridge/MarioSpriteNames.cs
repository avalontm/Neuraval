using System.Collections.Generic;

namespace Neuraval.Evolution.MarioBridge
{
    public static class MarioSpriteNames
    {
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
            { 0x29, "Koopa Kid (trono de Koopaling)" },
            { 0x2A, "Planta Piranha invertida" },
            { 0x2C, "Huevo de Yoshi" },
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
            { 0x92, "Splittin' Chuck" },
            { 0x93, "Bouncin' Chuck" },
            { 0x94, "Whistlin' Chuck" },
            { 0x95, "Clappin' Chuck" },
            { 0x97, "Puntin' Chuck" },
            { 0x98, "Pitchin' Chuck" },
            { 0x99, "Volcano Lotus" },
            { 0x9A, "Sumo Brother" },
            { 0x9B, "Hammer Brother" },
            { 0x9E, "Bola y cadena" },
            { 0x9F, "Banzai Bill" },
            { 0xA1, "Bola de Bowser" },
            { 0xA2, "Mecha-Koopa" },
            { 0xA6, "Hothead" },
            { 0xA8, "Blargg" },
            { 0xAA, "Fishbone" },
            { 0xAB, "Rex" },
            { 0xAC, "Pua de madera hacia abajo" },
            { 0xAD, "Pua de madera hacia arriba" },
            { 0xAE, "Fishin' Boo" },
            { 0xB2, "Pua cayendo" },
            { 0xB4, "Amoladora" },
            { 0xB9, "Caja de informacion (cartel de texto)" },
            { 0xBD, "Koopa deslizante" },
            { 0xBE, "Swooper" },
            { 0xBF, "Mega Mole" },
            { 0xC2, "Blurp (Cheep Cheep de castillo/agua)" },
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

        private static readonly HashSet<int> Hazardous = new()
        {
            0x00, 0x01, 0x02, 0x03, // Koopas sin caparazon
            0x04, 0x05, 0x06, 0x07, // Koopas
            0x08, 0x09, 0x0A, 0x0B, 0x0C, // Koopas voladores
            0x0D, // Bob-omb
            0x0F, 0x10, // Goombas
            0x11, 0x13, 0x14, // Buzzy Beetle y Spinies
            0x15, 0x16, 0x17, 0x18, // Cheep Cheeps
            0x1A, // Planta Piranha
            0x1C,
            0x1E, 0x1F, // Lakitu y Magikoopa
            0x22, 0x23, 0x24, 0x25, // Koopas de red
            0x28, // Boo
            0x2A, // Planta Piranha invertida
            0x2E, 0x30, 0x31, 0x32, // Spike Top y Dry Bones
            0x9F,
            0x44,
            0xA1,
            0x9E,
            0xB4,
            0x26,
            0x27,
            0x33,
            0x34,
            0x1D,
            0x50,
            0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D,
            0x3F, 0x40, 0x41, 0x42, 0x43,
            0x4D, 0x4E, 0x4F, 0x51,
            0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x86,
            0x91, 0x92, 0x93, 0x94, 0x95, 0x97, 0x98, 0x99, 0x9A, 0x9B,
            0x9E, 0xA2, 0xA6, 0xA8, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE,
            0xB2, 0xB4, 0xBD, 0xBE, 0xBF, 0xC2, 0xC3, 0xC5,
            0xDA, 0xDB, 0xDC, 0xDD, 0xDF
        };

        public static bool IsHazardous(int type)
        {
            return Hazardous.Contains(type);
        }

        public static bool IsKnown(int type)
        {
            return Known.ContainsKey(type);
        }
    }
}
