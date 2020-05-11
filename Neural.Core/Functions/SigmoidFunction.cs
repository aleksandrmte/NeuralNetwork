using System;
using System.Collections.Generic;
using System.Text;

namespace Neural.Core.Functions
{
    public class SigmoidFunction : IFunction
    {
        public double Calculate(double input)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -input));
            return result;
        }

        public double CalculateDx(double input)
        {
            var sigmoid = Calculate(input);
            var result = sigmoid / (1 - sigmoid);
            return result;
        }
    }
}
