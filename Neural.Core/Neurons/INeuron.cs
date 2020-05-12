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
        double Bias { get; set; }
        double BiasDelta { get; set; }
        double Gradient { get; set; }
        void SetInputs(List<double> inputs);
        void CalculateOutput();
        void Learn(double difference, double learningRate, double momentum);
    }
}
