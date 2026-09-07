using System;

namespace Neuraval.Core.Utils
{
    public class AdamMatrixOptimizer
    {
        private readonly float[,] _m;
        private readonly float[,] _v;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        private int _timeStep;

        public AdamMatrixOptimizer(int rows, int cols, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        {
            _m = new float[rows, cols];
            _v = new float[rows, cols];
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
            _timeStep = 0;
        }

        public void Update(float[,] parameters, float[,] gradients, float learningRate)
        {
            int rows = parameters.GetLength(0);
            int cols = parameters.GetLength(1);

            if (rows != _m.GetLength(0) || cols != _m.GetLength(1))
            {
                throw new ArgumentException(
                    $"Forma del parámetro ({rows}x{cols}) no coincide con el estado del optimizador ({_m.GetLength(0)}x{_m.GetLength(1)})");
            }

            _timeStep++;
            float biasCorrection1 = 1.0f - MathF.Pow(_beta1, _timeStep);
            float biasCorrection2 = 1.0f - MathF.Pow(_beta2, _timeStep);

            Matematicas.AdamUpdateAuto(parameters, gradients, _m, _v, _beta1, _beta2, _epsilon, learningRate, biasCorrection1, biasCorrection2);
        }

        public void Reset()
        {
            Matematicas.ParallelClearMatrix(_m);
            Matematicas.ParallelClearMatrix(_v);
            _timeStep = 0;
        }

        public AdamMatrixOptimizerState SaveState()
        {
            int rows = _m.GetLength(0);
            int cols = _m.GetLength(1);

            var state = new AdamMatrixOptimizerState
            {
                Rows = rows,
                Cols = cols,
                M = new float[rows * cols],
                V = new float[rows * cols],
                TimeStep = _timeStep
            };

            int index = 0;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    state.M[index] = _m[i, j];
                    state.V[index] = _v[i, j];
                    index++;
                }
            }

            return state;
        }

        public void LoadStateInto(AdamMatrixOptimizerState state)
        {
            int rows = _m.GetLength(0);
            int cols = _m.GetLength(1);

            if (state.Rows != rows || state.Cols != cols)
            {
                throw new ArgumentException(
                    $"Forma del estado del optimizador ({state.Rows}x{state.Cols}) no coincide con la forma actual ({rows}x{cols})");
            }

            int index = 0;
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    _m[i, j] = state.M[index];
                    _v[i, j] = state.V[index];
                    index++;
                }
            }

            _timeStep = state.TimeStep;
        }
    }

    public class AdamMatrixOptimizerState
    {
        public int Rows { get; set; }
        public int Cols { get; set; }
        public float[] M { get; set; } = Array.Empty<float>();
        public float[] V { get; set; } = Array.Empty<float>();
        public int TimeStep { get; set; }
    }

    public class AdamVectorOptimizer
    {
        private readonly float[] _m;
        private readonly float[] _v;
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        private int _timeStep;

        public AdamVectorOptimizer(int length, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        {
            _m = new float[length];
            _v = new float[length];
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
            _timeStep = 0;
        }

        public void Update(float[] parameters, float[] gradients, float learningRate)
        {
            if (parameters.Length != _m.Length || gradients.Length != _m.Length)
            {
                throw new ArgumentException(
                    $"Longitud del parámetro ({parameters.Length}) no coincide con el estado del optimizador ({_m.Length})");
            }

            _timeStep++;
            float biasCorrection1 = 1.0f - MathF.Pow(_beta1, _timeStep);
            float biasCorrection2 = 1.0f - MathF.Pow(_beta2, _timeStep);

            Matematicas.AdamUpdateAuto(parameters, gradients, _m, _v, _beta1, _beta2, _epsilon, learningRate, biasCorrection1, biasCorrection2);
        }

        public void Reset()
        {
            Array.Clear(_m, 0, _m.Length);
            Array.Clear(_v, 0, _v.Length);
            _timeStep = 0;
        }

        public AdamVectorOptimizerState SaveState()
        {
            return new AdamVectorOptimizerState
            {
                Length = _m.Length,
                M = (float[])_m.Clone(),
                V = (float[])_v.Clone(),
                TimeStep = _timeStep
            };
        }

        public void LoadStateInto(AdamVectorOptimizerState state)
        {
            if (state.Length != _m.Length)
            {
                throw new ArgumentException(
                    $"Longitud del estado del optimizador ({state.Length}) no coincide con la longitud actual ({_m.Length})");
            }

            Array.Copy(state.M, _m, _m.Length);
            Array.Copy(state.V, _v, _v.Length);
            _timeStep = state.TimeStep;
        }
    }

    public class AdamVectorOptimizerState
    {
        public int Length { get; set; }
        public float[] M { get; set; } = Array.Empty<float>();
        public float[] V { get; set; } = Array.Empty<float>();
        public int TimeStep { get; set; }
    }
}
