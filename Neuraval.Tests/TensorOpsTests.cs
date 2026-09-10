using System;
using Neuraval.Core.Utils;
using Neuraval.Tensor;
using Xunit;

namespace Neuraval.Tests
{
    public class TensorOpsTests
    {
        private const float Tolerance = 1e-4f;

        private static float[,] RandomMatrix(int rows, int cols, Random random)
        {
            var matrix = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                }
            }

            return matrix;
        }

        private static float[] RandomVector(int length, Random random)
        {
            var vector = new float[length];

            for (int i = 0; i < length; i++)
            {
                vector[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }

            return vector;
        }

        private static void AssertMatricesEqual(float[,] expected, float[,] actual)
        {
            Assert.Equal(expected.GetLength(0), actual.GetLength(0));
            Assert.Equal(expected.GetLength(1), actual.GetLength(1));

            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    Assert.True(MathF.Abs(expected[i, j] - actual[i, j]) < Tolerance,
                        $"Diferencia en [{i},{j}]: esperado {expected[i, j]}, obtenido {actual[i, j]}");
                }
            }
        }

        [Fact]
        public void MatMul_MatchesParallelMatrixMultiply()
        {
            var random = new Random(1);
            var a = RandomMatrix(6, 4, random);
            var b = RandomMatrix(4, 5, random);

            var expected = Matematicas.ParallelMatrixMultiply(a, b);

            var tensorA = Neuraval.Tensor.Tensor.FromArray2D(a);
            var tensorB = Neuraval.Tensor.Tensor.FromArray2D(b);
            var actual = TensorOps.MatMul(tensorA, tensorB).ToArray2D();

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void MatMul_WithIncompatibleShapes_Throws()
        {
            var a = new Neuraval.Tensor.Tensor(new[] { 2, 3 });
            var b = new Neuraval.Tensor.Tensor(new[] { 4, 2 });

            Assert.Throws<ArgumentException>(() => TensorOps.MatMul(a, b));
        }

        [Fact]
        public void MatMulTransposeB_MatchesParallelMatrixMultiplyTransposeB()
        {
            var random = new Random(2);
            var a = RandomMatrix(5, 4, random);
            var b = RandomMatrix(3, 4, random);

            var expected = Matematicas.ParallelMatrixMultiplyTransposeB(a, b, 0.5f);

            var tensorA = Neuraval.Tensor.Tensor.FromArray2D(a);
            var tensorB = Neuraval.Tensor.Tensor.FromArray2D(b);
            var actual = TensorOps.MatMulTransposeB(tensorA, tensorB, 0.5f).ToArray2D();

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void MatMulTransposeA_MatchesParallelMatrixMultiplyTransposeA()
        {
            var random = new Random(3);
            var a = RandomMatrix(4, 5, random);
            var b = RandomMatrix(4, 6, random);

            var expected = Matematicas.ParallelMatrixMultiplyTransposeA(a, b);

            var tensorA = Neuraval.Tensor.Tensor.FromArray2D(a);
            var tensorB = Neuraval.Tensor.Tensor.FromArray2D(b);
            var actual = TensorOps.MatMulTransposeA(tensorA, tensorB).ToArray2D();

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void Add_MatchesParallelMatrixAdd()
        {
            var random = new Random(4);
            var a = RandomMatrix(3, 3, random);
            var b = RandomMatrix(3, 3, random);

            var expected = Matematicas.ParallelMatrixAdd(a, b);

            var tensorA = Neuraval.Tensor.Tensor.FromArray2D(a);
            var tensorB = Neuraval.Tensor.Tensor.FromArray2D(b);
            var actual = TensorOps.Add(tensorA, tensorB).ToArray2D();

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void Add_WithMismatchedShapes_Throws()
        {
            var a = new Neuraval.Tensor.Tensor(new[] { 2, 2 });
            var b = new Neuraval.Tensor.Tensor(new[] { 3, 2 });

            Assert.Throws<ArgumentException>(() => TensorOps.Add(a, b));
        }

        [Fact]
        public void AddInPlace_MatchesParallelMatrixAddInPlace()
        {
            var random = new Random(5);
            var target = RandomMatrix(3, 3, random);
            var source = RandomMatrix(3, 3, random);
            var expected = (float[,])target.Clone();

            Matematicas.ParallelMatrixAddInPlace(expected, source, 0.25f);

            var tensorTarget = Neuraval.Tensor.Tensor.FromArray2D(target);
            var tensorSource = Neuraval.Tensor.Tensor.FromArray2D(source);
            TensorOps.AddInPlace(tensorTarget, tensorSource, 0.25f);

            AssertMatricesEqual(expected, tensorTarget.ToArray2D());
        }

        [Fact]
        public void Scale_MatchesParallelMatrixScale()
        {
            var random = new Random(6);
            var a = RandomMatrix(4, 4, random);

            var expected = Matematicas.ParallelMatrixScale(a, 3.5f);

            var tensorA = Neuraval.Tensor.Tensor.FromArray2D(a);
            var actual = TensorOps.Scale(tensorA, 3.5f).ToArray2D();

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void Transpose_MatchesParallelTranspose()
        {
            var random = new Random(7);
            var a = RandomMatrix(3, 5, random);

            var expected = Matematicas.ParallelTranspose(a);

            var tensorA = Neuraval.Tensor.Tensor.FromArray2D(a);
            var actual = TensorOps.Transpose(tensorA).ToArray2D();

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void ReLU_MatchesParallelReLU()
        {
            var random = new Random(8);
            var a = RandomMatrix(4, 4, random);

            var expected = Matematicas.ParallelReLU(a);

            var tensorA = Neuraval.Tensor.Tensor.FromArray2D(a);
            var actual = TensorOps.ReLU(tensorA).ToArray2D();

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void Sum_MatchesParallelSum()
        {
            var random = new Random(9);
            var a = RandomMatrix(5, 5, random);

            float expected = Matematicas.ParallelSum(a);

            var tensorA = Neuraval.Tensor.Tensor.FromArray2D(a);
            float actual = TensorOps.Sum(tensorA);

            Assert.True(MathF.Abs(expected - actual) < Tolerance);
        }

        [Fact]
        public void SoftmaxRows_MatchesParallelSoftmaxRows()
        {
            var random = new Random(10);
            var a = RandomMatrix(4, 6, random);

            var expected = Matematicas.ParallelSoftmaxRows(a);

            var tensorA = Neuraval.Tensor.Tensor.FromArray2D(a);
            var actual = TensorOps.SoftmaxRows(tensorA).ToArray2D();

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void ReLUBackward_MatchesManualDerivative()
        {
            var random = new Random(12);
            var activation = RandomMatrix(4, 5, random);
            var grad = RandomMatrix(4, 5, random);

            int rows = activation.GetLength(0);
            int cols = activation.GetLength(1);
            var expected = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    expected[i, j] = activation[i, j] > 0f ? grad[i, j] : 0f;
                }
            }

            var activationTensor = Neuraval.Tensor.Tensor.FromArray2D(activation);
            var gradTensor = Neuraval.Tensor.Tensor.FromArray2D(grad);
            var actual = TensorOps.ReLUBackward(gradTensor, activationTensor).ToArray2D();

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void ReLUBackward_WithMismatchedShapes_Throws()
        {
            var grad = new Neuraval.Tensor.Tensor(new[] { 2, 2 });
            var activation = new Neuraval.Tensor.Tensor(new[] { 3, 2 });

            Assert.Throws<ArgumentException>(() => TensorOps.ReLUBackward(grad, activation));
        }

        [Fact]
        public void Multiply_MatchesManualElementwiseProduct()
        {
            var random = new Random(13);
            var a = RandomMatrix(3, 4, random);
            var b = RandomMatrix(3, 4, random);

            int rows = a.GetLength(0);
            int cols = a.GetLength(1);
            var expected = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    expected[i, j] = a[i, j] * b[i, j];
                }
            }

            var tensorA = Neuraval.Tensor.Tensor.FromArray2D(a);
            var tensorB = Neuraval.Tensor.Tensor.FromArray2D(b);
            var actual = TensorOps.Multiply(tensorA, tensorB).ToArray2D();

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void Multiply_WithMismatchedShapes_Throws()
        {
            var a = new Neuraval.Tensor.Tensor(new[] { 2, 2 });
            var b = new Neuraval.Tensor.Tensor(new[] { 2, 3 });

            Assert.Throws<ArgumentException>(() => TensorOps.Multiply(a, b));
        }

        [Fact]
        public void SumRows_MatchesManualColumnSum()
        {
            var random = new Random(14);
            var a = RandomMatrix(5, 3, random);

            int rows = a.GetLength(0);
            int cols = a.GetLength(1);
            var expected = new float[cols];

            for (int j = 0; j < cols; j++)
            {
                float sum = 0f;
                for (int i = 0; i < rows; i++)
                {
                    sum += a[i, j];
                }
                expected[j] = sum;
            }

            var tensorA = Neuraval.Tensor.Tensor.FromArray2D(a);
            var actual = TensorOps.SumRows(tensorA).ToArray1D();

            for (int j = 0; j < cols; j++)
            {
                Assert.True(MathF.Abs(expected[j] - actual[j]) < Tolerance);
            }
        }

        [Fact]
        public void SumRows_WithRankOtherThanTwo_Throws()
        {
            var a = new Neuraval.Tensor.Tensor(new[] { 3 });

            Assert.Throws<ArgumentException>(() => TensorOps.SumRows(a));
        }

        [Fact]
        public void LayerNormRows_MatchesParallelLayerNormRows()
        {
            var random = new Random(11);
            var a = RandomMatrix(4, 6, random);
            var gamma = RandomVector(6, random);
            var beta = RandomVector(6, random);
            float epsilon = 1e-5f;

            var expected = Matematicas.ParallelLayerNormRows(a, gamma, beta, epsilon, out var expectedMean, out var expectedStd);

            var tensorA = Neuraval.Tensor.Tensor.FromArray2D(a);
            var tensorGamma = Neuraval.Tensor.Tensor.FromArray1D(gamma);
            var tensorBeta = Neuraval.Tensor.Tensor.FromArray1D(beta);
            var actual = TensorOps.LayerNormRows(tensorA, tensorGamma, tensorBeta, epsilon, out var actualMean, out var actualStd).ToArray2D();

            AssertMatricesEqual(expected, actual);

            for (int i = 0; i < expectedMean.Length; i++)
            {
                Assert.True(MathF.Abs(expectedMean[i] - actualMean[i]) < Tolerance);
                Assert.True(MathF.Abs(expectedStd[i] - actualStd[i]) < Tolerance);
            }
        }
    }
}
