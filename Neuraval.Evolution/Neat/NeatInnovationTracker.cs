using System.Collections.Generic;

namespace Neuraval.Evolution.Neat
{
    public sealed class NeatInnovationTracker
    {
        private int _nextNodeId;
        private int _nextInnovationNumber;
        private readonly Dictionary<(int InNode, int OutNode), int> _connectionInnovations = new();
        private readonly Dictionary<int, (int NodeId, int InnovationIn, int InnovationOut)> _nodeSplitInnovations = new();

        public NeatInnovationTracker(int startingNodeId)
        {
            _nextNodeId = startingNodeId;
            _nextInnovationNumber = 0;
        }

        public void FastForwardTo(int minNextNodeId, int minNextInnovationNumber)
        {
            if (minNextNodeId > _nextNodeId)
            {
                _nextNodeId = minNextNodeId;
            }

            if (minNextInnovationNumber > _nextInnovationNumber)
            {
                _nextInnovationNumber = minNextInnovationNumber;
            }
        }

        public int GetOrCreateConnectionInnovation(int inNode, int outNode)
        {
            var key = (inNode, outNode);

            if (_connectionInnovations.TryGetValue(key, out var existing))
            {
                return existing;
            }

            var innovation = _nextInnovationNumber++;
            _connectionInnovations[key] = innovation;
            return innovation;
        }

        public (int NodeId, int InnovationIn, int InnovationOut) SplitConnection(int connectionInnovation, int inNode, int outNode)
        {
            if (_nodeSplitInnovations.TryGetValue(connectionInnovation, out var existing))
            {
                return existing;
            }

            var nodeId = _nextNodeId++;
            var innovationIn = GetOrCreateConnectionInnovation(inNode, nodeId);
            var innovationOut = GetOrCreateConnectionInnovation(nodeId, outNode);
            var result = (nodeId, innovationIn, innovationOut);

            _nodeSplitInnovations[connectionInnovation] = result;
            return result;
        }
    }
}
