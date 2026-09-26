using System;
using System.Linq;

namespace Neuraval.Core.Serialization.SafeTensors
{
    public sealed class SafeTensorsEntry
    {
        public string Name { get; }

        public SafeTensorsDType DType { get; }

        public int[] Shape { get; }

        public float[] Data { get; }

        public SafeTensorsEntry(string name, SafeTensorsDType dtype, int[] shape, float[] data)
        {
            if (string.IsNullOrEmpty(name))
                throw new ArgumentException("El nombre del tensor no puede ser vacio", nameof(name));

            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            if (shape.Any(dim => dim <= 0))
                throw new ArgumentException("Todas las dimensiones del shape deben ser positivas", nameof(shape));

            if (data == null)
                throw new ArgumentNullException(nameof(data));

            int expectedLength = shape.Aggregate(1, (acc, dim) => acc * dim);
            if (expectedLength != data.Length)
                throw new ArgumentException(
                    $"El shape {string.Join('x', shape)} implica {expectedLength} elementos pero Data tiene {data.Length}");

            Name = name;
            DType = dtype;
            Shape = shape;
            Data = data;
        }
    }
}
