﻿using Neural.Core.Helpers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neural.Core.Functions
{
    public class SigmoidFunction : IFunction
    {
        public double Calculate(double input)
        {
            return input < -45.0 ? 0.0 : input > 45.0 ? 1.0 : 1.0 / (1.0 + Math.Exp(-input));
        }

        public double Derivation(double input)
        {
            return input * (1 - input);
        }
    }
}
