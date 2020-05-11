using System;
using System.Collections.Generic;

namespace Neural.Core.Neurons
{
    internal interface INeuron
    {
        Guid Id { get; set; }
        NeuronType Type { get; set; }
        List<double> Weights { get; set; }
        double Output { get; set; }
        double Delta { get; set; }
        void SetInputs(List<double> inputs);
        double CalculateOutput();
        void Learn(double difference, double learningRate);
    }
}
