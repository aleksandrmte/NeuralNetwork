using System;
using System.Collections.Generic;
using System.Text;

namespace Neural.Core.Functions
{
    public class SigmoidFunction : IFunction
    {
        public double Calculate(double x)
        {
            return x < -45.0 ? 0.0 : x > 45.0 ? 1.0 : 1.0 / (1.0 + Math.Exp(-x));
        }

        public double CalculateDx(double sigmoid)
        {
            return sigmoid * (1 - sigmoid);
        }

        //public double CalculateDx(double input)
        //{
        //    var sigmoid = Calculate(input);
        //    var result = sigmoid / (1 - sigmoid);
        //    return result;
        //}
    }
}
