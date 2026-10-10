using System;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class GqaGenerationCacheTests
    {
        private static float[,,,] RandomBlock(int batch, int seq, int heads, int headDim, int seed)
        {
            var rnd = new Random(seed);
            var block = new float[batch, seq, heads, headDim];
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < seq; s++)
                    for (int h = 0; h < heads; h++)
                        for (int d = 0; d < headDim; d++)
                            block[b, s, h, d] = (float)(rnd.NextDouble() * 2 - 1);
            return block;
        }

        [Theory]
        [InlineData(0, 4, 2, 2)]
        [InlineData(1, 0, 2, 2)]
        [InlineData(1, 4, 0, 2)]
        [InlineData(1, 4, 2, 0)]
        public void Constructor_InvalidDimensions_Throws(int batchSize, int capacity, int numKeyValueHeads, int headDim)
        {
            Assert.Throws<ArgumentException>(() => new GqaKeyValueCacheLayer(batchSize, capacity, numKeyValueHeads, headDim));
        }

        [Fact]
        public void Append_ThenGet_ReturnsExactValuesInOrder()
        {
            var cache = new GqaKeyValueCacheLayer(batchSize: 1, capacity: 5, numKeyValueHeads: 2, headDim: 3);
            var keysA = RandomBlock(1, 2, 2, 3, seed: 1);
            var valuesA = RandomBlock(1, 2, 2, 3, seed: 2);
            var keysB = RandomBlock(1, 1, 2, 3, seed: 3);
            var valuesB = RandomBlock(1, 1, 2, 3, seed: 4);

            cache.Append(keysA, valuesA);
            cache.Append(keysB, valuesB);

            var storedKeys = cache.GetKeys();
            var storedValues = cache.GetValues();

            Assert.Equal(3, cache.Length);

            for (int h = 0; h < 2; h++)
                for (int d = 0; d < 3; d++)
                {
                    Assert.Equal(keysA[0, 0, h, d], storedKeys[0, 0, h, d]);
                    Assert.Equal(keysA[0, 1, h, d], storedKeys[0, 1, h, d]);
                    Assert.Equal(keysB[0, 0, h, d], storedKeys[0, 2, h, d]);

                    Assert.Equal(valuesA[0, 0, h, d], storedValues[0, 0, h, d]);
                    Assert.Equal(valuesA[0, 1, h, d], storedValues[0, 1, h, d]);
                    Assert.Equal(valuesB[0, 0, h, d], storedValues[0, 2, h, d]);
                }
        }

        [Fact]
        public void Append_MultipleBatchesAndTokens_PreservesCacheStrides()
        {
            var cache = new GqaKeyValueCacheLayer(batchSize: 2, capacity: 4, numKeyValueHeads: 2, headDim: 2);
            var keys = RandomBlock(2, 2, 2, 2, seed: 11);
            var values = RandomBlock(2, 2, 2, 2, seed: 12);

            cache.Append(keys, values);

            var storedKeys = cache.GetKeys();
            var storedValues = cache.GetValues();
            for (int batch = 0; batch < 2; batch++)
                for (int token = 0; token < 2; token++)
                    for (int head = 0; head < 2; head++)
                        for (int dimension = 0; dimension < 2; dimension++)
                        {
                            Assert.Equal(keys[batch, token, head, dimension], storedKeys[batch, token, head, dimension]);
                            Assert.Equal(values[batch, token, head, dimension], storedValues[batch, token, head, dimension]);
                        }
        }

        [Fact]
        public void Append_ExceedingCapacity_Throws()
        {
            var cache = new GqaKeyValueCacheLayer(batchSize: 1, capacity: 2, numKeyValueHeads: 1, headDim: 2);
            var keys = RandomBlock(1, 3, 1, 2, seed: 1);
            var values = RandomBlock(1, 3, 1, 2, seed: 2);

            Assert.Throws<InvalidOperationException>(() => cache.Append(keys, values));
        }

        [Fact]
        public void Append_MismatchedBatchSize_Throws()
        {
            var cache = new GqaKeyValueCacheLayer(batchSize: 2, capacity: 4, numKeyValueHeads: 1, headDim: 2);
            var keys = RandomBlock(1, 1, 1, 2, seed: 1);
            var values = RandomBlock(1, 1, 1, 2, seed: 2);

            Assert.Throws<ArgumentException>(() => cache.Append(keys, values));
        }

        [Fact]
        public void Append_MismatchedHeadShape_Throws()
        {
            var cache = new GqaKeyValueCacheLayer(batchSize: 1, capacity: 4, numKeyValueHeads: 2, headDim: 2);
            var keys = RandomBlock(1, 1, 1, 2, seed: 1);
            var values = RandomBlock(1, 1, 1, 2, seed: 2);

            Assert.Throws<ArgumentException>(() => cache.Append(keys, values));
        }

        [Fact]
        public void Reset_ClearsLengthAndAllowsReappend()
        {
            var cache = new GqaKeyValueCacheLayer(batchSize: 1, capacity: 3, numKeyValueHeads: 1, headDim: 2);
            var keys = RandomBlock(1, 3, 1, 2, seed: 1);
            var values = RandomBlock(1, 3, 1, 2, seed: 2);

            cache.Append(keys, values);
            Assert.Equal(3, cache.Length);

            cache.Reset();
            Assert.Equal(0, cache.Length);

            cache.Append(keys, values);
            Assert.Equal(3, cache.Length);
        }

        [Fact]
        public void EstimatedMemoryBytes_MatchesExpectedFloatCount()
        {
            var cache = new GqaKeyValueCacheLayer(batchSize: 2, capacity: 10, numKeyValueHeads: 4, headDim: 8);

            long expected = 2L * 2 * 10 * 4 * 8 * sizeof(float);

            Assert.Equal(expected, cache.EstimatedMemoryBytes);
        }

        [Fact]
        public void GenerationCache_CreatesOneLayerPerRequestedCount()
        {
            var cache = new GqaGenerationCache(numLayers: 4, batchSize: 1, capacity: 8, numKeyValueHeads: 2, headDim: 4);

            Assert.Equal(4, cache.Layers.Count);
            Assert.Equal(0, cache.Length);
            Assert.Equal(8, cache.Capacity);
        }

        [Fact]
        public void GenerationCache_Reset_ResetsAllLayers()
        {
            var cache = new GqaGenerationCache(numLayers: 2, batchSize: 1, capacity: 4, numKeyValueHeads: 1, headDim: 2);
            var keys = RandomBlock(1, 2, 1, 2, seed: 1);
            var values = RandomBlock(1, 2, 1, 2, seed: 2);

            foreach (var layer in cache.Layers)
                layer.Append(keys, values);

            Assert.Equal(2, cache.Length);

            cache.Reset();

            Assert.Equal(0, cache.Length);
        }

        [Fact]
        public void GenerationCache_EstimatedMemoryBytes_SumsAllLayers()
        {
            var cache = new GqaGenerationCache(numLayers: 3, batchSize: 1, capacity: 5, numKeyValueHeads: 2, headDim: 4);

            long perLayer = 2L * 1 * 5 * 2 * 4 * sizeof(float);

            Assert.Equal(perLayer * 3, cache.EstimatedMemoryBytes);
        }

        [Fact]
        public void Constructor_NonPositiveNumLayers_Throws()
        {
            Assert.Throws<ArgumentException>(() => new GqaGenerationCache(0, 1, 4, 1, 2));
        }
    }
}
