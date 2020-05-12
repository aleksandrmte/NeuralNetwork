using Neural.Core.Functions;
using System;
using System.Collections.Generic;
using System.Linq;
using Neural.Core.Helpers;

namespace Neural.Core.Neurons
{
    internal class Neuron : INeuron
    {
        public IFunction Function;

        public Guid Id { get; set; }
        public List<double> Weights { get; set; }
        public List<double> WeightsDelta { get; set; }
        private List<double> Inputs { get; set; }
        public double Output { get; set; }
        public double Gradient { get; set; }
        public NeuronType Type { get; set; }
        public double Bias { get; set; }
        public double BiasDelta { get; set; }

        public Neuron(int inputCount, NeuronType type, IFunction function)
        {
            Function = function;
            Id = Guid.NewGuid();
            Type = type;
            Weights = new List<double>();
            WeightsDelta = new List<double>();
            Inputs = new List<double>();
            Bias = RandomHelper.GetRandom();

            InitInputRandomValue(inputCount);
        }

        private void InitInputRandomValue(int inputCount)
        {
            for (var i = 0; i < inputCount; i++)
            {
                WeightsDelta.Add(0);
                Weights.Add(RandomHelper.GetRandom());
            }
        }

        public void SetInputs(List<double> inputs)
        {
            Inputs = inputs;
        }

        public void CalculateOutput()
        {
            Output = Type == NeuronType.Input
                ? Inputs.First()
                : Function.Calculate(Inputs.Select((t, i) => t * Weights[i]).Sum() + Bias);
        }

        public void Learn(double difference, double learningRate, double momentum)
        {
            if (Type == NeuronType.Input)
                return;

            var prevDelta = BiasDelta;
            BiasDelta = learningRate * Gradient;
            Bias += BiasDelta + momentum * prevDelta;

            Gradient = difference * Function.Derivation(Output);

            for (var i = 0; i < Weights.Count; i++)
            {
                prevDelta = WeightsDelta[i];
                var input = Inputs[i];
                WeightsDelta[i] = input * Gradient * learningRate;
                Weights[i] += WeightsDelta[i] + momentum * prevDelta;
            }
        }
    }
}
