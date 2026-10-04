using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuraval.Samples.DinoGame.Sources
{
    public class ProbabilidadPorcentaje
    {
        private readonly Random random;

        public ProbabilidadPorcentaje() : this(Random.Shared)
        {
        }

        public ProbabilidadPorcentaje(Random random)
        {
            this.random = random ?? throw new ArgumentNullException(nameof(random));
        }

        public bool GenerarConProbabilidad(float porcentaje)
        {
            if (porcentaje < 0 || porcentaje > 100)
            {
                throw new ArgumentException("El porcentaje debe estar entre 0 y 100.");
            }

            int numeroAleatorio = random.Next(0, 101);

            return numeroAleatorio <= porcentaje;
        }
    }
}
