using Neural.Core.Functions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Neural.Core.Neurons
{
    internal class Neuron : INeuron
    {
        public IFunction Function;

        public Guid Id { get; set; }
        public List<double> Weights { get; set; }
        private List<double> Inputs { get; set; }
        public double Output { get; set; }
        public double Delta { get; set; }
        public NeuronType Type { get; set; }

        public Neuron(int inputCount, NeuronType type, IFunction function)
        {
            Function = function;
            Id = Guid.NewGuid();
            Type = type;
            Weights = new List<double>();
            Inputs = new List<double>();

            InitInputRandomValue(inputCount);
        }

        private void InitInputRandomValue(int inputCount)
        {
            var rnd = new Random();
            for (var i = 0; i < inputCount; i++)
            {
                if (Type == NeuronType.Input)
                    Weights.Add(1);
                else
                    Weights.Add(rnd.NextDouble());
            }
        }

        public void SetInputs(List<double> inputs)
        {
            Inputs = inputs;
        }

        public double CalculateOutput()
        {
            var sum = Inputs.Select((t, i) => t * Weights[i]).Sum();

            Output = Type != NeuronType.Input ? Function.Calculate(sum) : sum;

            return Output;
        }

        public void Learn(double difference, double learningRate)
        {
            if (Type == NeuronType.Input)
                return;

            Delta = difference * Function.CalculateDx(Output);

            for (var i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }
    }
}
