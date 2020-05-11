using System;
using Neural.Core.Functions;
using Neural.Core.Layers;
using Neural.Core.Neurons;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Neural.Core.Helpers;

namespace Neural.Core
{
    public sealed class NeuralNetwork
    {
        private readonly List<Layer> _layers;
        private readonly double _learningRate;

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

        public void ShowAll()
        {
            var lastLayer = _layers.Last();
            Console.WriteLine(string.Join("; ", lastLayer.Neurons.Select(x => x.Output)));
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
                neuron.SetInputs(new List<double> { signals[i] });
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

        public double Train(double[] expectedResults, double[,] inputs, int countEpoch)
        {
            var error = 0.0;
            for (var i = 0; i < countEpoch; i++)
            {
                for (var j = 0; j < expectedResults.Length; j++)
                {
                    var output = expectedResults[j];
                    var input = ArrayHelper.GetRow(inputs, j);
                    error += BackPropagation(output, input.ToList());
                }
            }
            var result = error / countEpoch;
            return result;
        }
        
        public double[,] Scaling(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];
            for (var column = 0; column < inputs.GetLength(1); column++)
            {
                var columnData = ArrayHelper.GetColumn(inputs, column);
                var min = columnData.OrderBy(x => x).First();
                var max = columnData.OrderByDescending(x => x).First();
                ChangeScalingColumnData(result, inputs, column, min, max);
            }
            return result;
        }

        private static void ChangeScalingColumnData(double[,] result, double[,] inputs, int column, double min, double max)
        {
            var denominator = max - min;
            for (var row = 0; row < inputs.GetLength(0); row++)
            {
                result[row, column] = (inputs[row, column] - min) / denominator;
            }
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
