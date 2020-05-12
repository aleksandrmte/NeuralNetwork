using System;
using System.Collections.Generic;
using System.Text;

namespace Neural.Core.Functions
{
    public class ThFunction : IFunction
    {
        public double Calculate(double input)
        {
            var x = Math.Pow(Math.E, input);
            var y = Math.Pow(Math.E, -input);
            return (x - y) / (x + y);
        }

        public double Derivation(double input)
        {
            return 1 - Math.Pow(input, 2);
        }
    }
}
