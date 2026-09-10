using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuraval.Evolution.Neat
{
    public sealed class NeatGenome
    {
        public int InputCount { get; }
        public int OutputCount { get; }
        public List<NeatNodeGene> Nodes { get; }
        public List<NeatConnectionGene> Connections { get; }

        public NeatGenome(int inputCount, int outputCount, List<NeatNodeGene> nodes, List<NeatConnectionGene> connections)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            Nodes = nodes;
            Connections = connections;
        }

        public static NeatGenome CreateInitial(int inputCount, int outputCount, Random random, NeatInnovationTracker tracker)
        {
            var nodes = new List<NeatNodeGene>();

            for (var i = 0; i < inputCount; i++)
            {
                nodes.Add(new NeatNodeGene(i, NeatNodeType.Input));
            }

            var biasId = inputCount;
            nodes.Add(new NeatNodeGene(biasId, NeatNodeType.Bias));

            var outputStart = biasId + 1;
            for (var i = 0; i < outputCount; i++)
            {
                nodes.Add(new NeatNodeGene(outputStart + i, NeatNodeType.Output));
            }

            var connections = new List<NeatConnectionGene>();
            for (var i = 0; i < inputCount + 1; i++)
            {
                var inNode = i < inputCount ? i : biasId;

                for (var j = 0; j < outputCount; j++)
                {
                    var outNode = outputStart + j;
                    var innovation = tracker.GetOrCreateConnectionInnovation(inNode, outNode);
                    var weight = random.NextSingle() * 2f - 1f;
                    connections.Add(new NeatConnectionGene(inNode, outNode, weight, true, innovation));
                }
            }

            return new NeatGenome(inputCount, outputCount, nodes, connections);
        }

        public NeatGenome Clone()
        {
            return new NeatGenome(
                InputCount,
                OutputCount,
                Nodes.Select(node => node.Clone()).ToList(),
                Connections.Select(connection => connection.Clone()).ToList());
        }

        private List<int>? _cachedOrder;
        private Dictionary<int, NeatNodeGene>? _cachedNodesById;
        private Dictionary<int, List<NeatConnectionGene>>? _cachedIncoming;
        private bool _structureDirty = true;

        private void InvalidateStructureCache()
        {
            _structureDirty = true;
        }

        private void EnsureStructureCache()
        {
            if (!_structureDirty && _cachedOrder != null)
            {
                return;
            }

            var nodesById = new Dictionary<int, NeatNodeGene>(Nodes.Count);
            var adjacency = new Dictionary<int, List<int>>(Nodes.Count);

            foreach (var node in Nodes)
            {
                nodesById[node.Id] = node;
                adjacency[node.Id] = new List<int>();
            }

            var incoming = new Dictionary<int, List<NeatConnectionGene>>();

            foreach (var connection in Connections)
            {
                if (!connection.Enabled)
                {
                    continue;
                }

                adjacency[connection.InNode].Add(connection.OutNode);

                if (!incoming.TryGetValue(connection.OutNode, out var list))
                {
                    list = new List<NeatConnectionGene>();
                    incoming[connection.OutNode] = list;
                }

                list.Add(connection);
            }

            var visited = new HashSet<int>();
            var order = new List<int>(Nodes.Count);

            void Visit(int nodeId)
            {
                if (!visited.Add(nodeId))
                {
                    return;
                }

                foreach (var next in adjacency[nodeId])
                {
                    Visit(next);
                }

                order.Add(nodeId);
            }

            foreach (var node in Nodes)
            {
                Visit(node.Id);
            }

            order.Reverse();

            _cachedNodesById = nodesById;
            _cachedIncoming = incoming;
            _cachedOrder = order;
            _structureDirty = false;
        }

        public float[] Evaluate(float[] inputs)
        {
            EnsureStructureCache();

            var values = new Dictionary<int, float>(_cachedOrder!.Count);

            for (var i = 0; i < InputCount; i++)
            {
                values[i] = inputs[i];
            }

            values[InputCount] = 1f;

            foreach (var nodeId in _cachedOrder)
            {
                var node = _cachedNodesById![nodeId];

                if (node.Type == NeatNodeType.Input || node.Type == NeatNodeType.Bias)
                {
                    continue;
                }

                var sum = 0f;

                if (_cachedIncoming!.TryGetValue(nodeId, out var connectionsIn))
                {
                    foreach (var connection in connectionsIn)
                    {
                        if (values.TryGetValue(connection.InNode, out var inputValue))
                        {
                            sum += inputValue * connection.Weight;
                        }
                    }
                }

                values[nodeId] = node.Type == NeatNodeType.Output ? Sigmoid(sum) : MathF.Tanh(sum);
            }

            var outputs = new float[OutputCount];
            var outputStart = InputCount + 1;

            for (var i = 0; i < OutputCount; i++)
            {
                values.TryGetValue(outputStart + i, out var value);
                outputs[i] = value;
            }

            return outputs;
        }

        private static float Sigmoid(float value)
        {
            return 1f / (1f + MathF.Exp(-value));
        }

        public void MutateWeights(Random random, float perturbRate, float perturbStrength, float resetRate)
        {
            foreach (var connection in Connections)
            {
                var roll = random.NextDouble();

                if (roll < resetRate)
                {
                    connection.Weight = random.NextSingle() * 2f - 1f;
                }
                else if (roll < resetRate + perturbRate)
                {
                    connection.Weight += random.NextGaussian(0f, perturbStrength);
                }
            }
        }

        public bool TryMutateAddConnection(Random random, NeatInnovationTracker tracker, int maxAttempts)
        {
            for (var attempt = 0; attempt < maxAttempts; attempt++)
            {
                var a = Nodes[random.Next(Nodes.Count)];
                var b = Nodes[random.Next(Nodes.Count)];

                if (a.Id == b.Id)
                {
                    continue;
                }

                if (b.Type == NeatNodeType.Input || b.Type == NeatNodeType.Bias)
                {
                    continue;
                }

                if (a.Type == NeatNodeType.Output && b.Type == NeatNodeType.Output)
                {
                    continue;
                }

                if (Connections.Any(connection => connection.InNode == a.Id && connection.OutNode == b.Id))
                {
                    continue;
                }

                if (CreatesCycle(a.Id, b.Id))
                {
                    continue;
                }

                var innovation = tracker.GetOrCreateConnectionInnovation(a.Id, b.Id);
                var weight = random.NextSingle() * 2f - 1f;
                Connections.Add(new NeatConnectionGene(a.Id, b.Id, weight, true, innovation));
                InvalidateStructureCache();
                return true;
            }

            return false;
        }

        public void MutateAddNode(Random random, NeatInnovationTracker tracker)
        {
            var enabledConnections = Connections.Where(connection => connection.Enabled).ToList();

            if (enabledConnections.Count == 0)
            {
                return;
            }

            var target = enabledConnections[random.Next(enabledConnections.Count)];
            target.Enabled = false;

            var (nodeId, innovationIn, innovationOut) = tracker.SplitConnection(target.Innovation, target.InNode, target.OutNode);

            Nodes.Add(new NeatNodeGene(nodeId, NeatNodeType.Hidden));
            Connections.Add(new NeatConnectionGene(target.InNode, nodeId, 1f, true, innovationIn));
            Connections.Add(new NeatConnectionGene(nodeId, target.OutNode, target.Weight, true, innovationOut));
            InvalidateStructureCache();
        }

        private bool CreatesCycle(int fromId, int toId)
        {
            if (fromId == toId)
            {
                return true;
            }

            var adjacency = Nodes.ToDictionary(node => node.Id, _ => new List<int>());

            foreach (var connection in Connections)
            {
                if (connection.Enabled)
                {
                    adjacency[connection.InNode].Add(connection.OutNode);
                }
            }

            adjacency[fromId].Add(toId);

            var visited = new HashSet<int>();
            var stack = new Stack<int>();
            stack.Push(toId);

            while (stack.Count > 0)
            {
                var current = stack.Pop();

                if (current == fromId)
                {
                    return true;
                }

                if (!visited.Add(current))
                {
                    continue;
                }

                foreach (var next in adjacency[current])
                {
                    stack.Push(next);
                }
            }

            return false;
        }

        public static NeatGenome Crossover(NeatGenome fitter, NeatGenome other, Random random)
        {
            var otherByInnovation = other.Connections.ToDictionary(connection => connection.Innovation);
            var offspringConnections = new List<NeatConnectionGene>();

            foreach (var gene in fitter.Connections)
            {
                if (otherByInnovation.TryGetValue(gene.Innovation, out var matching))
                {
                    var chosen = random.NextDouble() < 0.5 ? gene : matching;
                    var enabled = true;

                    if (!gene.Enabled || !matching.Enabled)
                    {
                        enabled = random.NextDouble() >= 0.75;
                    }

                    offspringConnections.Add(new NeatConnectionGene(chosen.InNode, chosen.OutNode, chosen.Weight, enabled, chosen.Innovation));
                }
                else
                {
                    offspringConnections.Add(gene.Clone());
                }
            }

            var referencedNodeIds = new HashSet<int>();
            foreach (var connection in offspringConnections)
            {
                referencedNodeIds.Add(connection.InNode);
                referencedNodeIds.Add(connection.OutNode);
            }

            var nodesByFitter = fitter.Nodes.ToDictionary(node => node.Id);
            var nodesByOther = other.Nodes.ToDictionary(node => node.Id);

            var offspringNodes = new List<NeatNodeGene>();
            for (var i = 0; i < fitter.InputCount; i++)
            {
                offspringNodes.Add(new NeatNodeGene(i, NeatNodeType.Input));
            }

            offspringNodes.Add(new NeatNodeGene(fitter.InputCount, NeatNodeType.Bias));

            for (var i = 0; i < fitter.OutputCount; i++)
            {
                offspringNodes.Add(new NeatNodeGene(fitter.InputCount + 1 + i, NeatNodeType.Output));
            }

            var baseNodeIds = new HashSet<int>(offspringNodes.Select(node => node.Id));

            foreach (var nodeId in referencedNodeIds)
            {
                if (baseNodeIds.Contains(nodeId))
                {
                    continue;
                }

                if (nodesByFitter.TryGetValue(nodeId, out var node))
                {
                    offspringNodes.Add(node.Clone());
                }
                else if (nodesByOther.TryGetValue(nodeId, out var otherNode))
                {
                    offspringNodes.Add(otherNode.Clone());
                }
            }

            return new NeatGenome(fitter.InputCount, fitter.OutputCount, offspringNodes, offspringConnections);
        }

        public static float CompatibilityDistance(NeatGenome a, NeatGenome b, float excessCoefficient, float disjointCoefficient, float weightCoefficient)
        {
            var aByInnovation = a.Connections.ToDictionary(connection => connection.Innovation);
            var bByInnovation = b.Connections.ToDictionary(connection => connection.Innovation);

            var maxInnovationA = a.Connections.Count == 0 ? 0 : a.Connections.Max(connection => connection.Innovation);
            var maxInnovationB = b.Connections.Count == 0 ? 0 : b.Connections.Max(connection => connection.Innovation);
            var lowerMax = Math.Min(maxInnovationA, maxInnovationB);

            var matching = 0;
            var disjoint = 0;
            var excess = 0;
            var weightDifferenceSum = 0f;

            foreach (var innovation in aByInnovation.Keys.Union(bByInnovation.Keys))
            {
                var inA = aByInnovation.TryGetValue(innovation, out var geneA);
                var inB = bByInnovation.TryGetValue(innovation, out var geneB);

                if (inA && inB)
                {
                    matching++;
                    weightDifferenceSum += MathF.Abs(geneA!.Weight - geneB!.Weight);
                }
                else if (innovation > lowerMax)
                {
                    excess++;
                }
                else
                {
                    disjoint++;
                }
            }

            var n = Math.Max(a.Connections.Count, b.Connections.Count);
            n = n < 20 ? 1 : n;

            var averageWeightDifference = matching == 0 ? 0f : weightDifferenceSum / matching;

            return excessCoefficient * excess / n + disjointCoefficient * disjoint / n + weightCoefficient * averageWeightDifference;
        }
    }
}
