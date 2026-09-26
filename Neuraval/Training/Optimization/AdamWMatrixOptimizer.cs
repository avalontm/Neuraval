using System;

namespace Neuraval.Core.Training.Optimization
{
    public sealed class AdamWMatrixOptimizer
    {
        private readonly float[,] _m;
        private readonly float[,] _v;
        private readonly AdamWOptions _options;
        private int _timeStep;

        public AdamWMatrixOptimizer(int rows, int cols, AdamWOptions? options = null)
        {
            if (rows <= 0)
                throw new ArgumentOutOfRangeException(nameof(rows), "rows debe ser mayor que cero");

            if (cols <= 0)
                throw new ArgumentOutOfRangeException(nameof(cols), "cols debe ser mayor que cero");

            _m = new float[rows, cols];
            _v = new float[rows, cols];
            _options = options ?? AdamWOptions.Create();
        }

        public AdamWOptions Options => _options;

        public int TimeStep => _timeStep;

        public void Update(float[,] parameters, float[,] gradients)
        {
            Update(parameters, gradients, _options.LearningRate);
        }

        public void Update(float[,] parameters, float[,] gradients, float learningRate)
        {
            int rows = _m.GetLength(0);
            int cols = _m.GetLength(1);

            if (parameters.GetLength(0) != rows || parameters.GetLength(1) != cols)
                throw new ArgumentException(
                    $"Forma del parámetro ({parameters.GetLength(0)}x{parameters.GetLength(1)}) no coincide con el estado del optimizador ({rows}x{cols})");

            if (gradients.GetLength(0) != rows || gradients.GetLength(1) != cols)
                throw new ArgumentException(
                    $"Forma del gradiente ({gradients.GetLength(0)}x{gradients.GetLength(1)}) no coincide con el estado del optimizador ({rows}x{cols})");

            _timeStep++;
            float biasCorrection1 = 1f - MathF.Pow(_options.Beta1, _timeStep);
            float biasCorrection2 = 1f - MathF.Pow(_options.Beta2, _timeStep);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float g = gradients[i, j];

                    _m[i, j] = _options.Beta1 * _m[i, j] + (1f - _options.Beta1) * g;
                    _v[i, j] = _options.Beta2 * _v[i, j] + (1f - _options.Beta2) * g * g;

                    float mHat = _m[i, j] / biasCorrection1;
                    float vHat = _v[i, j] / biasCorrection2;

                    float adaptiveStep = mHat / (MathF.Sqrt(vHat) + _options.Epsilon);
                    float decayStep = _options.WeightDecay * parameters[i, j];

                    parameters[i, j] -= learningRate * (adaptiveStep + decayStep);
                }
            }
        }

        public void Reset()
        {
            Array.Clear(_m, 0, _m.Length);
            Array.Clear(_v, 0, _v.Length);
            _timeStep = 0;
        }

        public AdamWMatrixOptimizerState SaveState()
        {
            int rows = _m.GetLength(0);
            int cols = _m.GetLength(1);

            var state = new AdamWMatrixOptimizerState
            {
                Rows = rows,
                Cols = cols,
                M = new float[rows * cols],
                V = new float[rows * cols],
                TimeStep = _timeStep
            };

            Buffer.BlockCopy(_m, 0, state.M, 0, rows * cols * sizeof(float));
            Buffer.BlockCopy(_v, 0, state.V, 0, rows * cols * sizeof(float));

            return state;
        }

        public void LoadStateInto(AdamWMatrixOptimizerState state)
        {
            int rows = _m.GetLength(0);
            int cols = _m.GetLength(1);

            if (state.Rows != rows || state.Cols != cols)
                throw new ArgumentException(
                    $"Forma del estado del optimizador ({state.Rows}x{state.Cols}) no coincide con la forma actual ({rows}x{cols})");

            Buffer.BlockCopy(state.M, 0, _m, 0, rows * cols * sizeof(float));
            Buffer.BlockCopy(state.V, 0, _v, 0, rows * cols * sizeof(float));

            _timeStep = state.TimeStep;
        }
    }

    public sealed class AdamWMatrixOptimizerState
    {
        public int Rows { get; set; }
        public int Cols { get; set; }
        public float[] M { get; set; } = Array.Empty<float>();
        public float[] V { get; set; } = Array.Empty<float>();
        public int TimeStep { get; set; }
    }
}
