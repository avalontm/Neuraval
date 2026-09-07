using System.IO;

namespace Neuraval.Evolution.MarioBridge
{
    using Neuraval.Evolution.Neat;

    public static class MarioGenomeSerializer
    {
        public static void Write(BinaryWriter writer, NeatGenome genome)
        {
            writer.Write(genome.InputCount);
            writer.Write(genome.OutputCount);

            writer.Write(genome.Nodes.Count);
            foreach (var node in genome.Nodes)
            {
                writer.Write(node.Id);
                writer.Write((byte)node.Type);
            }

            writer.Write(genome.Connections.Count);
            foreach (var connection in genome.Connections)
            {
                writer.Write(connection.InNode);
                writer.Write(connection.OutNode);
                writer.Write(connection.Weight);
                writer.Write(connection.Enabled);
                writer.Write(connection.Innovation);
            }
        }

        public static NeatGenome Read(BinaryReader reader)
        {
            var inputCount = reader.ReadInt32();
            var outputCount = reader.ReadInt32();

            var nodeCount = reader.ReadInt32();
            var nodes = new List<NeatNodeGene>(nodeCount);
            for (var i = 0; i < nodeCount; i++)
            {
                var id = reader.ReadInt32();
                var type = (NeatNodeType)reader.ReadByte();
                nodes.Add(new NeatNodeGene(id, type));
            }

            var connectionCount = reader.ReadInt32();
            var connections = new List<NeatConnectionGene>(connectionCount);
            for (var i = 0; i < connectionCount; i++)
            {
                var inNode = reader.ReadInt32();
                var outNode = reader.ReadInt32();
                var weight = reader.ReadSingle();
                var enabled = reader.ReadBoolean();
                var innovation = reader.ReadInt32();
                connections.Add(new NeatConnectionGene(inNode, outNode, weight, enabled, innovation));
            }

            return new NeatGenome(inputCount, outputCount, nodes, connections);
        }
    }
}