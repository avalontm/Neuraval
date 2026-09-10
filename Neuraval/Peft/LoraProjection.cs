using System;
using Neuraval.Core.Utils;
using Neuraval.Cuda;
using Neuraval.Tensor;

namespace Neuraval.Core.Models
{
    /// <summary>
    /// Adaptador LoRA para una única proyección lineal congelada W (in x out).
    ///
    /// En vez de actualizar W directamente, LoRA agrega una corrección de bajo
    /// rango: y = x·W + scaling·(x·A)·B, con A (in x rank) y B (rank x out),
    /// rank &lt;&lt; min(in, out). Solo A y B se entrenan (cada uno con su propio
    /// Adam); W nunca se toca desde aquí — quien decide si W se congela es
    /// <see cref="Models.MultiHeadAttention"/> a través de su propio flag.
    ///
    /// B se inicializa en cero (A se inicializa aleatoria) para que el
    /// adaptador arranque siendo la identidad (no altera el modelo base hasta
    /// que empieza a entrenar), que es la inicialización estándar de LoRA.
    /// </summary>
    public class LoraProjection
    {
        private readonly int _inDim;
        private readonly int _outDim;
        private readonly int _rank;
        private readonly float _alpha;
        private readonly float _scaling;

        private float[,] _matrixA;
        private float[,] _matrixB;

        private Neuraval.Tensor.Tensor _gradA;
        private Neuraval.Tensor.Tensor _gradB;
        private Neuraval.Tensor.Tensor _accumulatedGradA;
        private Neuraval.Tensor.Tensor _accumulatedGradB;

        private AdamMatrixOptimizer _optimizerA;
        private AdamMatrixOptimizer _optimizerB;

        private CudaWeightCache _cacheA;
        private CudaWeightCache _cacheB;

        private Neuraval.Tensor.Tensor? _lastInput;
        private Neuraval.Tensor.Tensor? _lastXa;

        public int InDim => _inDim;
        public int OutDim => _outDim;
        public int Rank => _rank;
        public float Alpha => _alpha;

        public LoraProjection(int inDim, int outDim, int rank, float alpha, int seed)
        {
            if (rank <= 0)
            {
                throw new ArgumentException("El rank de LoRA debe ser positivo");
            }

            _inDim = inDim;
            _outDim = outDim;
            _rank = rank;
            _alpha = alpha;
            _scaling = alpha / rank;

            var random = new Random(seed);
            float limit = MathF.Sqrt(6.0f / (inDim + rank));

            _matrixA = new float[inDim, rank];
            for (int i = 0; i < inDim; i++)
            {
                for (int j = 0; j < rank; j++)
                {
                    _matrixA[i, j] = (random.NextSingle() * 2 - 1) * limit;
                }
            }

            // B arranca en cero: el adaptador no cambia nada hasta que se entrena.
            _matrixB = new float[rank, outDim];

            _gradA = new Neuraval.Tensor.Tensor(new[] { inDim, rank });
            _gradB = new Neuraval.Tensor.Tensor(new[] { rank, outDim });
            _accumulatedGradA = new Neuraval.Tensor.Tensor(new[] { inDim, rank });
            _accumulatedGradB = new Neuraval.Tensor.Tensor(new[] { rank, outDim });

            _optimizerA = new AdamMatrixOptimizer(inDim, rank);
            _optimizerB = new AdamMatrixOptimizer(rank, outDim);

            _cacheA = new CudaWeightCache(inDim, rank);
            _cacheB = new CudaWeightCache(rank, outDim);
        }

        /// <summary>
        /// Calcula la corrección de bajo rango ya escalada (scaling·(x·A)·B) y
        /// cachea las tensores intermedios necesarios para <see cref="Backward"/>.
        /// El resultado se suma directo al output de la proyección base.
        /// </summary>
        public Neuraval.Tensor.Tensor Forward(Neuraval.Tensor.Tensor inputTensor, DeviceType device)
        {
            _lastInput = inputTensor;

            var xa = TensorOps.MatMulCachedB(inputTensor, _matrixA, _cacheA);
            _lastXa = xa;

            var delta = TensorOps.MatMulCachedB(xa, _matrixB, _cacheB);
            return TensorOps.Scale(delta, _scaling);
        }

        /// <summary>
        /// Recibe el gradiente respecto al output de la proyección (el mismo que
        /// recibe la rama base) y devuelve la contribución de LoRA al gradiente
        /// de entrada. Acumula los gradientes de A y B internamente.
        /// </summary>
        public Neuraval.Tensor.Tensor Backward(Neuraval.Tensor.Tensor gradOutputTensor, DeviceType device)
        {
            if (_lastInput == null || _lastXa == null)
            {
                throw new InvalidOperationException("Forward must be called before Backward");
            }

            var gradUnscaled = TensorOps.Scale(gradOutputTensor, _scaling);

            var gradBTensor = TensorOps.MatMulTransposeA(_lastXa, gradUnscaled);
            var gradXaTensor = TensorOps.MatMulTransposeBCachedB(gradUnscaled, _matrixB, _cacheB);
            var gradATensor = TensorOps.MatMulTransposeA(_lastInput, gradXaTensor);
            var gradInputTensor = TensorOps.MatMulTransposeBCachedB(gradXaTensor, _matrixA, _cacheA);

            TensorOps.AddInPlace(_accumulatedGradA, gradATensor);
            TensorOps.AddInPlace(_accumulatedGradB, gradBTensor);

            return gradInputTensor;
        }

        public void ZeroGradients()
        {
            TensorOps.Clear(_accumulatedGradA);
            TensorOps.Clear(_accumulatedGradB);
        }

        public void AverageGradients(int batchSize)
        {
            float scale = 1.0f / batchSize;

            _gradA = TensorOps.Scale(_accumulatedGradA, scale);
            _gradB = TensorOps.Scale(_accumulatedGradB, scale);
        }

        public void ScaleGradients(float scale)
        {
            _gradA = TensorOps.Scale(_gradA, scale);
            _gradB = TensorOps.Scale(_gradB, scale);
        }

        private static float SumSquared(Neuraval.Tensor.Tensor tensor)
        {
            return TensorOps.Sum(TensorOps.Multiply(tensor, tensor));
        }

        public float SumSquaredGradients()
        {
            return SumSquared(_gradA) + SumSquared(_gradB);
        }

        public void UpdateWeights(float learningRate)
        {
            _optimizerA.Update(_matrixA, _gradA.ToArray2D(), learningRate);
            _optimizerB.Update(_matrixB, _gradB.ToArray2D(), learningRate);

            _cacheA.Invalidate();
            _cacheB.Invalidate();

            TensorOps.Clear(_gradA);
            TensorOps.Clear(_gradB);
        }

        public void ResetGradients()
        {
            TensorOps.Clear(_gradA);
            TensorOps.Clear(_gradB);
            TensorOps.Clear(_accumulatedGradA);
            TensorOps.Clear(_accumulatedGradB);
        }

        public LoraProjectionState SaveState()
        {
            return new LoraProjectionState
            {
                InDim = _inDim,
                OutDim = _outDim,
                Rank = _rank,
                Alpha = _alpha,
                MatrixA = Flatten(_matrixA),
                MatrixB = Flatten(_matrixB),
                OptimizerAState = _optimizerA.SaveState(),
                OptimizerBState = _optimizerB.SaveState()
            };
        }

        public static LoraProjection LoadState(LoraProjectionState state)
        {
            var projection = new LoraProjection(state.InDim, state.OutDim, state.Rank, state.Alpha, seed: 1);

            projection._matrixA = Unflatten(state.MatrixA, state.InDim, state.Rank);
            projection._matrixB = Unflatten(state.MatrixB, state.Rank, state.OutDim);

            if (state.OptimizerAState != null) projection._optimizerA.LoadStateInto(state.OptimizerAState);
            if (state.OptimizerBState != null) projection._optimizerB.LoadStateInto(state.OptimizerBState);

            projection._cacheA.Invalidate();
            projection._cacheB.Invalidate();

            return projection;
        }

        private static float[] Flatten(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows * cols];

            // float[,] rectangular es contiguo row-major: aplanar es un memcpy
            // puro, no una copia elemento a elemento.
            System.Buffer.BlockCopy(matrix, 0, result, 0, result.Length * sizeof(float));

            return result;
        }

        private static float[,] Unflatten(float[] array, int rows, int cols)
        {
            var matrix = new float[rows, cols];

            System.Buffer.BlockCopy(array, 0, matrix, 0, array.Length * sizeof(float));

            return matrix;
        }
    }

    /// <summary>Estado serializable de un <see cref="LoraProjection"/> individual.</summary>
    public class LoraProjectionState
    {
        public int InDim { get; set; }
        public int OutDim { get; set; }
        public int Rank { get; set; }
        public float Alpha { get; set; }
        public float[] MatrixA { get; set; } = Array.Empty<float>();
        public float[] MatrixB { get; set; } = Array.Empty<float>();
        public AdamMatrixOptimizerState? OptimizerAState { get; set; }
        public AdamMatrixOptimizerState? OptimizerBState { get; set; }
    }

    /// <summary>
    /// Estado serializable de los cuatro adaptadores LoRA de una capa de
    /// atención (Q/K/V/O), más el flag de si los pesos base de esa capa
    /// quedaron congelados durante el fine-tuning.
    /// </summary>
    public class LoraAttentionState
    {
        public LoraProjectionState Query { get; set; } = new LoraProjectionState();
        public LoraProjectionState Key { get; set; } = new LoraProjectionState();
        public LoraProjectionState Value { get; set; } = new LoraProjectionState();
        public LoraProjectionState Output { get; set; } = new LoraProjectionState();
        public bool FreezeBaseWeights { get; set; }
    }
}
