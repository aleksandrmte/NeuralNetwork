using Neural.Core.Neurons;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neural.Core.Layers
{
    internal class Layer
    {
        public Layer()
        {
            Neurons = new List<INeuron>();
        }

        public List<INeuron> Neurons { get; set; }

        public List<double> GetOutputSignals()
        {
            var result = new List<double>();
            foreach(var item in Neurons)
            {
                result.Add(item.Output);
            }
            return result;
        }
    }
}
