using System;

namespace Neuraval.Core.Training.Optimization
{
    public sealed class AdamWVectorOptimizer
    {
        private readonly float[] _m;
        private readonly float[] _v;
        private readonly AdamWOptions _options;
        private int _timeStep;

        public AdamWVectorOptimizer(int length, AdamWOptions? options = null)
        {
            if (length <= 0)
                throw new ArgumentOutOfRangeException(nameof(length), "length debe ser mayor que cero");

            _m = new float[length];
            _v = new float[length];
            _options = options ?? AdamWOptions.Create();
        }

        public AdamWOptions Options => _options;

        public int TimeStep => _timeStep;

        public void Update(float[] parameters, float[] gradients)
        {
            Update(parameters, gradients, _options.LearningRate);
        }

        public void Update(float[] parameters, float[] gradients, float learningRate)
        {
            if (parameters.Length != _m.Length)
                throw new ArgumentException(
                    $"Longitud del parámetro ({parameters.Length}) no coincide con el estado del optimizador ({_m.Length})");

            if (gradients.Length != _m.Length)
                throw new ArgumentException(
                    $"Longitud del gradiente ({gradients.Length}) no coincide con el estado del optimizador ({_m.Length})");

            _timeStep++;
            float biasCorrection1 = 1f - MathF.Pow(_options.Beta1, _timeStep);
            float biasCorrection2 = 1f - MathF.Pow(_options.Beta2, _timeStep);

            for (int i = 0; i < parameters.Length; i++)
            {
                float g = gradients[i];

                _m[i] = _options.Beta1 * _m[i] + (1f - _options.Beta1) * g;
                _v[i] = _options.Beta2 * _v[i] + (1f - _options.Beta2) * g * g;

                float mHat = _m[i] / biasCorrection1;
                float vHat = _v[i] / biasCorrection2;

                float adaptiveStep = mHat / (MathF.Sqrt(vHat) + _options.Epsilon);
                float decayStep = _options.WeightDecay * parameters[i];

                parameters[i] -= learningRate * (adaptiveStep + decayStep);
            }
        }

        public void Reset()
        {
            Array.Clear(_m, 0, _m.Length);
            Array.Clear(_v, 0, _v.Length);
            _timeStep = 0;
        }

        public AdamWVectorOptimizerState SaveState()
        {
            return new AdamWVectorOptimizerState
            {
                Length = _m.Length,
                M = (float[])_m.Clone(),
                V = (float[])_v.Clone(),
                TimeStep = _timeStep
            };
        }

        public void LoadStateInto(AdamWVectorOptimizerState state)
        {
            if (state.Length != _m.Length)
                throw new ArgumentException(
                    $"Longitud del estado del optimizador ({state.Length}) no coincide con la longitud actual ({_m.Length})");

            Array.Copy(state.M, _m, _m.Length);
            Array.Copy(state.V, _v, _v.Length);
            _timeStep = state.TimeStep;
        }
    }

    public sealed class AdamWVectorOptimizerState
    {
        public int Length { get; set; }
        public float[] M { get; set; } = Array.Empty<float>();
        public float[] V { get; set; } = Array.Empty<float>();
        public int TimeStep { get; set; }
    }
}
