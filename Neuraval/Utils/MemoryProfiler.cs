using System.Collections.Generic;

namespace Neuraval.Core.Utils
{
    public struct MemoryComponentReport
    {
        public string Name;
        public long ParameterCount;
        public long WeightsBytes;
        public long GradientBytes;
        public long OptimizerStateBytes;

        public long TotalBytes => WeightsBytes + GradientBytes + OptimizerStateBytes;
    }

    public struct ActivationEstimate
    {
        public string Name;
        public long ElementCount;
        public long Bytes;
    }

    public class ModelMemoryReport
    {
        public List<MemoryComponentReport> Parameters { get; } = new List<MemoryComponentReport>();
        public List<ActivationEstimate> Activations { get; } = new List<ActivationEstimate>();

        public long TotalParameterCount
        {
            get
            {
                long sum = 0;
                foreach (var component in Parameters)
                {
                    sum += component.ParameterCount;
                }
                return sum;
            }
        }

        public long TotalWeightsBytes
        {
            get
            {
                long sum = 0;
                foreach (var component in Parameters)
                {
                    sum += component.WeightsBytes;
                }
                return sum;
            }
        }

        public long TotalGradientBytes
        {
            get
            {
                long sum = 0;
                foreach (var component in Parameters)
                {
                    sum += component.GradientBytes;
                }
                return sum;
            }
        }

        public long TotalOptimizerStateBytes
        {
            get
            {
                long sum = 0;
                foreach (var component in Parameters)
                {
                    sum += component.OptimizerStateBytes;
                }
                return sum;
            }
        }

        public long TotalParameterBytes => TotalWeightsBytes + TotalGradientBytes + TotalOptimizerStateBytes;

        public long TotalActivationBytes
        {
            get
            {
                long sum = 0;
                foreach (var activation in Activations)
                {
                    sum += activation.Bytes;
                }
                return sum;
            }
        }

        public long GrandTotalBytes => TotalParameterBytes + TotalActivationBytes;
    }

    public static class MemoryProfiler
    {
        private const int FloatBytes = 4;

        public static ModelMemoryReport Analyze(
            int vocabSize,
            int embeddingDim,
            int numLayers,
            int numHeads,
            int feedforwardDim,
            int batchSize,
            int sequenceLength)
        {
            var report = new ModelMemoryReport();

            report.Parameters.Add(MakeParameterComponent("Embedding", (long)vocabSize * embeddingDim));

            long attentionParamsPerBlock = 4L * embeddingDim * embeddingDim;
            report.Parameters.Add(MakeParameterComponent("Atención Q/K/V/O", attentionParamsPerBlock * numLayers));

            long feedforwardParamsPerBlock = (long)embeddingDim * feedforwardDim + feedforwardDim
                + (long)feedforwardDim * embeddingDim + embeddingDim;
            report.Parameters.Add(MakeParameterComponent("FeedForward W1/b1/W2/b2", feedforwardParamsPerBlock * numLayers));

            long layerNormParamsPerBlock = 4L * embeddingDim;
            report.Parameters.Add(MakeParameterComponent("LayerNorm (2 por bloque)", layerNormParamsPerBlock * numLayers));

            report.Parameters.Add(MakeParameterComponent("LayerNorm final", 2L * embeddingDim));
            report.Parameters.Add(MakeParameterComponent("Output bias", vocabSize));

            report.Activations.Add(MakeActivation("Hidden states", (long)(numLayers + 2) * sequenceLength * embeddingDim * batchSize));
            report.Activations.Add(MakeActivation("Q/K/V/concat", 4L * sequenceLength * embeddingDim * numLayers * batchSize));
            report.Activations.Add(MakeActivation("Attention scores por cabeza", (long)numHeads * sequenceLength * sequenceLength * numLayers * batchSize));
            report.Activations.Add(MakeActivation("FeedForward hidden", (long)sequenceLength * feedforwardDim * numLayers * batchSize));
            report.Activations.Add(MakeActivation("Logits", (long)sequenceLength * vocabSize * batchSize));

            return report;
        }

        private static MemoryComponentReport MakeParameterComponent(string name, long parameterCount)
        {
            long weightsBytes = parameterCount * FloatBytes;
            return new MemoryComponentReport
            {
                Name = name,
                ParameterCount = parameterCount,
                WeightsBytes = weightsBytes,
                GradientBytes = weightsBytes * 2,
                OptimizerStateBytes = weightsBytes * 2
            };
        }

        private static ActivationEstimate MakeActivation(string name, long elementCount)
        {
            return new ActivationEstimate
            {
                Name = name,
                ElementCount = elementCount,
                Bytes = elementCount * FloatBytes
            };
        }

        public static string FormatBytes(long bytes)
        {
            double kb = bytes / 1024.0;
            double mb = kb / 1024.0;
            double gb = mb / 1024.0;

            if (gb >= 1.0)
            {
                return $"{gb:F2} GB";
            }

            if (mb >= 1.0)
            {
                return $"{mb:F2} MB";
            }

            if (kb >= 1.0)
            {
                return $"{kb:F2} KB";
            }

            return $"{bytes} B";
        }
    }
}
