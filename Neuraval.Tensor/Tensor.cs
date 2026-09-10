using System;

namespace Neuraval.Tensor
{
    public sealed class Tensor
    {
        public float[] Buffer { get; }
        public int[] Shape { get; }
        public int[] Strides { get; }
        public DeviceType Device { get; }
        public DType DType { get; }
        public int Length { get; }

        public int Rank => Shape.Length;

        public Tensor(int[] shape, DeviceType device = DeviceType.Cpu, DType dtype = DType.Float32)
        {
            Shape = ValidateAndCloneShape(shape);
            Strides = ComputeStrides(Shape);
            Length = ComputeLength(Shape);
            Buffer = new float[Length];
            Device = device;
            DType = dtype;
        }

        public Tensor(float[] buffer, int[] shape, DeviceType device = DeviceType.Cpu, DType dtype = DType.Float32)
        {
            Shape = ValidateAndCloneShape(shape);
            Strides = ComputeStrides(Shape);
            Length = ComputeLength(Shape);

            if (buffer.Length != Length)
            {
                throw new ArgumentException($"El buffer tiene {buffer.Length} elementos pero el shape requiere {Length}");
            }

            Buffer = buffer;
            Device = device;
            DType = dtype;
        }

        private static int[] ValidateAndCloneShape(int[] shape)
        {
            if (shape == null || shape.Length == 0)
            {
                throw new ArgumentException("El shape debe tener al menos una dimensión");
            }

            foreach (int dim in shape)
            {
                if (dim <= 0)
                {
                    throw new ArgumentException("Todas las dimensiones del shape deben ser positivas");
                }
            }

            return (int[])shape.Clone();
        }

        private static int[] ComputeStrides(int[] shape)
        {
            var strides = new int[shape.Length];
            int stride = 1;

            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }

            return strides;
        }

        private static int ComputeLength(int[] shape)
        {
            long length = 1;

            foreach (int dim in shape)
            {
                length *= dim;
            }

            if (length > int.MaxValue)
            {
                throw new ArgumentException("El tensor excede el tamaño máximo soportado por un array de .NET");
            }

            return (int)length;
        }

        private int ComputeFlatIndex(int[] indices)
        {
            if (indices.Length != Shape.Length)
            {
                throw new ArgumentException($"Se esperaban {Shape.Length} índices pero se recibieron {indices.Length}");
            }

            int flatIndex = 0;

            for (int i = 0; i < indices.Length; i++)
            {
                if (indices[i] < 0 || indices[i] >= Shape[i])
                {
                    throw new ArgumentOutOfRangeException(nameof(indices), $"El índice {indices[i]} está fuera de rango para la dimensión {i} (tamaño {Shape[i]})");
                }

                flatIndex += indices[i] * Strides[i];
            }

            return flatIndex;
        }

        public float this[params int[] indices]
        {
            get => Buffer[ComputeFlatIndex(indices)];
            set => Buffer[ComputeFlatIndex(indices)] = value;
        }

        public Tensor Reshape(params int[] newShape)
        {
            int newLength = ComputeLength(newShape);

            if (newLength != Length)
            {
                throw new ArgumentException($"No se puede reinterpretar un tensor de {Length} elementos con un shape de {newLength} elementos");
            }

            return new Tensor(Buffer, newShape, Device, DType);
        }

        public Tensor Clone()
        {
            var copy = new float[Buffer.Length];
            Array.Copy(Buffer, copy, Buffer.Length);
            return new Tensor(copy, Shape, Device, DType);
        }

        public bool ShapeEquals(Tensor other)
        {
            if (Shape.Length != other.Shape.Length)
            {
                return false;
            }

            for (int i = 0; i < Shape.Length; i++)
            {
                if (Shape[i] != other.Shape[i])
                {
                    return false;
                }
            }

            return true;
        }

        public static Tensor Zeros(params int[] shape)
        {
            return new Tensor(shape);
        }

        public static Tensor Ones(params int[] shape)
        {
            var tensor = new Tensor(shape);

            for (int i = 0; i < tensor.Buffer.Length; i++)
            {
                tensor.Buffer[i] = 1.0f;
            }

            return tensor;
        }

        public static Tensor Full(float value, params int[] shape)
        {
            var tensor = new Tensor(shape);

            for (int i = 0; i < tensor.Buffer.Length; i++)
            {
                tensor.Buffer[i] = value;
            }

            return tensor;
        }

        public static Tensor FromArray1D(float[] source, DeviceType device = DeviceType.Cpu)
        {
            var buffer = new float[source.Length];
            Array.Copy(source, buffer, source.Length);
            return new Tensor(buffer, new[] { source.Length }, device);
        }

        public static Tensor FromArray2D(float[,] source, DeviceType device = DeviceType.Cpu)
        {
            int rows = source.GetLength(0);
            int cols = source.GetLength(1);
            var tensor = new Tensor(new[] { rows, cols }, device);

            // float[,] rectangular es contiguo row-major en memoria: es el mismo
            // layout que el Buffer plano del tensor, así que es un memcpy puro
            // (mismo criterio que TransformerModel.FlattenBatch / AdamMatrixOptimizer.SaveState).
            System.Buffer.BlockCopy(source, 0, tensor.Buffer, 0, tensor.Length * sizeof(float));

            return tensor;
        }

        public static Tensor FromArray3D(float[,,] source, DeviceType device = DeviceType.Cpu)
        {
            int d0 = source.GetLength(0);
            int d1 = source.GetLength(1);
            int d2 = source.GetLength(2);
            var tensor = new Tensor(new[] { d0, d1, d2 }, device);

            System.Buffer.BlockCopy(source, 0, tensor.Buffer, 0, tensor.Length * sizeof(float));

            return tensor;
        }

        public float[] ToArray1D()
        {
            if (Rank != 1)
            {
                throw new InvalidOperationException($"ToArray1D requiere rank 1, el tensor tiene rank {Rank}");
            }

            var result = new float[Shape[0]];
            Array.Copy(Buffer, result, Buffer.Length);
            return result;
        }

        public float[,] ToArray2D()
        {
            if (Rank != 2)
            {
                throw new InvalidOperationException($"ToArray2D requiere rank 2, el tensor tiene rank {Rank}");
            }

            int rows = Shape[0];
            int cols = Shape[1];
            var result = new float[rows, cols];

            System.Buffer.BlockCopy(this.Buffer, 0, result, 0, Length * sizeof(float));

            return result;
        }

        public float[,,] ToArray3D()
        {
            if (Rank != 3)
            {
                throw new InvalidOperationException($"ToArray3D requiere rank 3, el tensor tiene rank {Rank}");
            }

            int d0 = Shape[0];
            int d1 = Shape[1];
            int d2 = Shape[2];
            var result = new float[d0, d1, d2];

            System.Buffer.BlockCopy(this.Buffer, 0, result, 0, Length * sizeof(float));

            return result;
        }
    }
}
