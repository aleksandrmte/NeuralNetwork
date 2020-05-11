using Neural.Core.Functions;
using Neural.Core.Neurons;

namespace Neural.Core.Layers
{
    internal class LayerFactory
    {
        public Layer CreateLayer(int inputCount, int neuronCount, NeuronType type, IFunction function)
        {
            var layer = new Layer();
            for (var i = 0; i < neuronCount; i++)
            {
                layer.Neurons.Add(new Neuron(inputCount, type, function));
            }
            return layer;
        }
    }
}
