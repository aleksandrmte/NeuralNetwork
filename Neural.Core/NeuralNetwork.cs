using Neural.Core.Functions;
using Neural.Core.Layers;
using Neural.Core.Neurons;
using System.Collections.Generic;
using System.Linq;

namespace Neural.Core
{
    public sealed class NeuralNetwork
    {
        private readonly List<Layer> _layers;
        private readonly double _learningRate;
        private List<double> _expectedResults;

        public NeuralNetwork(double learningRate, IFunction function, params int[] countLayerNeurons)
        {
            _layers = new List<Layer>();
            _learningRate = learningRate;
            var factory = new LayerFactory();
            for (var i = 0; i < countLayerNeurons.Length; i++)
            {
                var type = i == 0 ? NeuronType.Input : i == countLayerNeurons.Length - 1 ? NeuronType.Output : NeuronType.Normal;
                var countInputs = i == 0 ? 1 : countLayerNeurons[i - 1];
                var layer = factory.CreateLayer(countInputs, countLayerNeurons[i], type, function);
                _layers.Add(layer);
            }
        }

        public double Activate(List<double> signals)
        {
            SendInputSignals(signals);
            SendSignalsToAllLayers();
            return GetOutput();
        }

        private double GetOutput()
        {
            var lastLayer = _layers.Last();
            return lastLayer.Neurons.OrderByDescending(n => n.Output).First().Output;
        }

        private void SendInputSignals(IReadOnlyList<double> signals)
        {
            var firstLayer = _layers.First();
            for (var i = 0; i < firstLayer.Neurons.Count; i++)
            {
                var neuron = firstLayer.Neurons[i];
                neuron.SetInputs(new List<double> {signals[i]});
                neuron.CalculateOutput();
            }            
        }

        private void SendSignalsToAllLayers()
        {
            for (var i = 1; i < _layers.Count; i++)
            {
                var layer = _layers[i];
                var previousLayerSignals = _layers[i - 1].GetOutputSignals();
                foreach (var neuron in layer.Neurons)
                {
                    neuron.SetInputs(previousLayerSignals);
                    neuron.CalculateOutput();
                }
            }
        }

        public void SetExpectedResults(List<double> expectedResults)
        {
            _expectedResults = expectedResults;
        }

        public double Train(int countEpoch, List<List<double>> dataSets)
        {
            var error = 0.0;
            for (var i = 0; i < countEpoch; i++)
            {
                error += dataSets.Select((dataSet, j) => BackPropagation(_expectedResults[j], dataSet)).Sum();
            }
            var result = error / countEpoch;
            return result;
        }

        private double BackPropagation(double expectedValue, List<double> inputs)
        {
            var outputValue = Activate(inputs);
            var difference = outputValue - expectedValue;

            HandleOutputLayer(difference);
            HandleHiddenLayers();

            var result = difference * difference;
            return result;
        }

        private void HandleOutputLayer(double difference)
        {
            var layer = _layers.Last();
            foreach (var neuron in layer.Neurons)
            {
                neuron.Learn(difference, _learningRate);
            }
        }

        private void HandleHiddenLayers()
        {
            for (var i = _layers.Count - 2; i >= 0; i--)
            {
                var layer = _layers[i];
                var higherLayer = _layers[i + 1];

                for (var j = 0; j < layer.Neurons.Count; j++)
                {
                    var neuron = layer.Neurons[j];

                    foreach (var previousNeuron in higherLayer.Neurons)
                    {
                        var error = previousNeuron.Weights[j] * previousNeuron.Delta;
                        neuron.Learn(error, _learningRate);
                    }
                }
            }
        }
    }
}
