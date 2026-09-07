using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuraval.Samples.DinoGame
{
    public static class Extenciones
    {
        public static double NextDouble(this Random random,double minimum, double maximum)
        {
            return random.NextDouble() * (maximum - minimum) + minimum;
        }
    }
}
