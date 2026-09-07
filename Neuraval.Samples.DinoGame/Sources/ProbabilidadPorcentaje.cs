using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuraval.Samples.DinoGame.Sources
{
    public class ProbabilidadPorcentaje
    {
        private Random random;

        public ProbabilidadPorcentaje()
        {
            random = new Random();
        }

        public bool GenerarConProbabilidad(float porcentaje)
        {
            if (porcentaje < 0 || porcentaje > 100)
            {
                throw new ArgumentException("El porcentaje debe estar entre 0 y 100.");
            }

            int numeroAleatorio = random.Next(0, 101); // Genera un número aleatorio entre 0 y 100

            return numeroAleatorio <= porcentaje;
        }
    }
}
