using System;
using System.Collections.Generic;
using System.Text;

namespace Neural.Core.Functions
{
    public interface IFunction
    {
        double Calculate(double input);
        double Derivation(double input);
    }
}
