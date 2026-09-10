using System;
using Neuraval.Tensor;
using Xunit;

namespace Neuraval.Tests
{
    public class TensorTests
    {
        [Fact]
        public void Constructor_WithShape_AllocatesZeroedBuffer()
        {
            var tensor = new Neuraval.Tensor.Tensor(new[] { 2, 3 });

            Assert.Equal(6, tensor.Length);
            Assert.Equal(new[] { 2, 3 }, tensor.Shape);

            foreach (float value in tensor.Buffer)
            {
                Assert.Equal(0.0f, value);
            }
        }

        [Fact]
        public void Constructor_ComputesRowMajorStrides()
        {
            var tensor = new Neuraval.Tensor.Tensor(new[] { 2, 3, 4 });

            Assert.Equal(new[] { 12, 4, 1 }, tensor.Strides);
        }

        [Fact]
        public void Constructor_DefaultsToCpuAndFloat32()
        {
            var tensor = new Neuraval.Tensor.Tensor(new[] { 2, 2 });

            Assert.Equal(DeviceType.Cpu, tensor.Device);
            Assert.Equal(DType.Float32, tensor.DType);
        }

        [Fact]
        public void Constructor_WithMismatchedBufferLength_Throws()
        {
            var buffer = new float[5];

            Assert.Throws<ArgumentException>(() => new Neuraval.Tensor.Tensor(buffer, new[] { 2, 3 }));
        }

        [Fact]
        public void Constructor_WithNonPositiveDimension_Throws()
        {
            Assert.Throws<ArgumentException>(() => new Neuraval.Tensor.Tensor(new[] { 2, 0 }));
        }

        [Fact]
        public void Indexer_GetAndSet_UsesRowMajorLayout()
        {
            var tensor = new Neuraval.Tensor.Tensor(new[] { 2, 3 });

            tensor[0, 0] = 1.0f;
            tensor[0, 2] = 2.0f;
            tensor[1, 1] = 3.0f;

            Assert.Equal(1.0f, tensor[0, 0]);
            Assert.Equal(2.0f, tensor[0, 2]);
            Assert.Equal(3.0f, tensor[1, 1]);
            Assert.Equal(2.0f, tensor.Buffer[2]);
            Assert.Equal(3.0f, tensor.Buffer[4]);
        }

        [Fact]
        public void Indexer_OutOfRange_Throws()
        {
            var tensor = new Neuraval.Tensor.Tensor(new[] { 2, 2 });

            Assert.Throws<ArgumentOutOfRangeException>(() => tensor[2, 0]);
        }

        [Fact]
        public void Indexer_WrongRank_Throws()
        {
            var tensor = new Neuraval.Tensor.Tensor(new[] { 2, 2 });

            Assert.Throws<ArgumentException>(() => tensor[0]);
        }

        [Fact]
        public void Ones_FillsAllElementsWithOne()
        {
            var tensor = Neuraval.Tensor.Tensor.Ones(2, 2);

            foreach (float value in tensor.Buffer)
            {
                Assert.Equal(1.0f, value);
            }
        }

        [Fact]
        public void Full_FillsAllElementsWithGivenValue()
        {
            var tensor = Neuraval.Tensor.Tensor.Full(7.5f, 3, 2);

            foreach (float value in tensor.Buffer)
            {
                Assert.Equal(7.5f, value);
            }
        }

        [Fact]
        public void Reshape_PreservesBufferAndElementOrder()
        {
            var tensor = new Neuraval.Tensor.Tensor(new[] { 2, 3 });

            for (int i = 0; i < tensor.Buffer.Length; i++)
            {
                tensor.Buffer[i] = i;
            }

            var reshaped = tensor.Reshape(3, 2);

            Assert.Equal(new[] { 3, 2 }, reshaped.Shape);
            Assert.Same(tensor.Buffer, reshaped.Buffer);
            Assert.Equal(4.0f, reshaped[2, 0]);
        }

        [Fact]
        public void Reshape_WithIncompatibleElementCount_Throws()
        {
            var tensor = new Neuraval.Tensor.Tensor(new[] { 2, 3 });

            Assert.Throws<ArgumentException>(() => tensor.Reshape(4, 2));
        }

        [Fact]
        public void Clone_ProducesIndependentBuffer()
        {
            var tensor = Neuraval.Tensor.Tensor.Ones(2, 2);
            var clone = tensor.Clone();

            clone[0, 0] = 99.0f;

            Assert.Equal(1.0f, tensor[0, 0]);
            Assert.Equal(99.0f, clone[0, 0]);
            Assert.NotSame(tensor.Buffer, clone.Buffer);
        }

        [Fact]
        public void ShapeEquals_ComparesDimensionsNotBuffer()
        {
            var a = Neuraval.Tensor.Tensor.Zeros(2, 3);
            var b = Neuraval.Tensor.Tensor.Ones(2, 3);
            var c = Neuraval.Tensor.Tensor.Zeros(3, 2);

            Assert.True(a.ShapeEquals(b));
            Assert.False(a.ShapeEquals(c));
        }

        [Fact]
        public void FromArray2D_AndToArray2D_RoundTrip()
        {
            var source = new float[,] { { 1, 2, 3 }, { 4, 5, 6 } };

            var tensor = Neuraval.Tensor.Tensor.FromArray2D(source);
            var result = tensor.ToArray2D();

            Assert.Equal(new[] { 2, 3 }, tensor.Shape);

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(source[i, j], result[i, j]);
                }
            }
        }

        [Fact]
        public void FromArray3D_AndToArray3D_RoundTrip()
        {
            var source = new float[2, 2, 2];
            int counter = 0;

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        source[i, j, k] = counter++;
                    }
                }
            }

            var tensor = Neuraval.Tensor.Tensor.FromArray3D(source);
            var result = tensor.ToArray3D();

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        Assert.Equal(source[i, j, k], result[i, j, k]);
                    }
                }
            }
        }

        [Fact]
        public void FromArray1D_AndToArray1D_RoundTrip()
        {
            var source = new float[] { 1.5f, 2.5f, 3.5f };

            var tensor = Neuraval.Tensor.Tensor.FromArray1D(source);
            var result = tensor.ToArray1D();

            Assert.Equal(source, result);
        }

        [Fact]
        public void ToArray2D_WithWrongRank_Throws()
        {
            var tensor = new Neuraval.Tensor.Tensor(new[] { 2, 2, 2 });

            Assert.Throws<InvalidOperationException>(() => tensor.ToArray2D());
        }

        [Fact]
        public void FromArray2D_ProducesIndependentBufferFromSource()
        {
            var source = new float[,] { { 1, 2 }, { 3, 4 } };
            var tensor = Neuraval.Tensor.Tensor.FromArray2D(source);

            source[0, 0] = 999;

            Assert.Equal(1.0f, tensor[0, 0]);
        }
    }
}
