namespace Neuraval.Core.Utils
{
    public static class FuncionesActivacion
    {
        public static float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        public static float SigmoidDerivative(float x)
        {
            return x * (1.0f - x);
        }

        public static float Tanh(float x)
        {
            return (float)Math.Tanh(x);
        }

        public static float TanhDerivative(float x)
        {
            return 1.0f - (x * x);
        }

        public static float ReLU(float x)
        {
            return Math.Max(0, x);
        }

        public static float ReLUDerivative(float x)
        {
            return x > 0 ? 1.0f : 0.0f;
        }

        public static float LeakyReLU(float x, float alpha = 0.01f)
        {
            return x > 0 ? x : alpha * x;
        }

        public static float LeakyReLUDerivative(float x, float alpha = 0.01f)
        {
            return x > 0 ? 1.0f : alpha;
        }

        public static float Linear(float x)
        {
            return x;
        }

        public static float LinearDerivative(float x)
        {
            return 1.0f;
        }

        public static float[] Softmax(float[] values)
        {
            float max = values.Max();
            float[] exp = values.Select(v => (float)Math.Exp(v - max)).ToArray();
            float sum = exp.Sum();
            return exp.Select(e => e / sum).ToArray();
        }
    }
}