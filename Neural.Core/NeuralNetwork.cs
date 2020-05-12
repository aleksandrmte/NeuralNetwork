using Neural.Core.Functions;
using Neural.Core.Helpers;
using Neural.Core.Layers;
using Neural.Core.Neurons;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Neural.Core
{
    public sealed class NeuralNetwork
    {
        private readonly List<Layer> _layers;
        private readonly double _learningRate;
        private double _momentum;
        public List<double> Errors { get; private set; }

        public NeuralNetwork(double learningRate, IFunction function, params int[] countLayerNeurons)
        {
            _layers = new List<Layer>();
            Errors = new List<double>();
            _learningRate = learningRate;
            _momentum = 0;
            var factory = new LayerFactory();
            for (var i = 0; i < countLayerNeurons.Length; i++)
            {
                var type = i == 0 ? NeuronType.Input : i == countLayerNeurons.Length - 1 ? NeuronType.Output : NeuronType.Normal;
                var countInputs = i == 0 ? 1 : countLayerNeurons[i - 1];
                var layer = factory.CreateLayer(countInputs, countLayerNeurons[i], type, function);
                _layers.Add(layer);
            }
        }

        public void SetMomentum(double momentum)
        {
            _momentum = momentum;
        }

        public List<double> Activate(List<double> signals)
        {
            SendInputSignals(signals);
            SendSignalsToAllLayers();
            return GetOutput();
        }

        private List<double> GetOutput()
        {
            var lastLayer = _layers.Last();
            return lastLayer.Neurons.Select(x => x.Output).ToList();
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

        public double Train(double[,] expectedResults, double[,] inputs, int countEpoch)
        {
            var error = 0.0;
            for (var i = 0; i < countEpoch; i++)
            {
                var errorEpoch = 0.0;
                for (var j = 0; j < expectedResults.GetLength(0); j++)
                {
                    var output = ArrayHelper.GetRow(expectedResults, j);
                    var input = ArrayHelper.GetRow(inputs, j);
                    errorEpoch += BackPropagation(output.ToList(), input.ToList());                    
                }
                error += errorEpoch;
                Errors.Add(errorEpoch);
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

        private double BackPropagation(List<double> expectedValues, List<double> inputs)
        {
            var outputValues = Activate(inputs);
            var result = HandleOutputLayer(expectedValues, outputValues);
            HandleHiddenLayers();
            result = Math.Pow(result, 2);
            return result;
        }

        private double HandleOutputLayer(List<double> expectedValues, List<double> outputValues)
        {
            var result = 0.0;
            var lastLayer = _layers.Last();
            var lastLayerNeurons = lastLayer.Neurons;
            var i = 0;
            foreach (var expectedValue in expectedValues)
            {
                var difference = expectedValue - outputValues[i];
                var neuron = lastLayerNeurons[i];
                neuron.Learn(difference, _learningRate, _momentum);
                result += difference;
                i++;
            }
            return result;
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
                    var error = 0.0;
                    foreach (var neuronHigherLayer in higherLayer.Neurons)
                    {
                        error += neuronHigherLayer.Weights[j] * neuronHigherLayer.Gradient;
                    }
                    neuron.Learn(error, _learningRate, _momentum);
                }
            }
        }
    }
}
