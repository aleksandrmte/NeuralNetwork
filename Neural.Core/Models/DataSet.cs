using System;
using System.Collections.Generic;
using System.Text;

namespace Neural.Core.Models
{
    public class DataSet
    {
        public double[,] Input { get; set; }
        public double[,] Output { get; set; }
    }
}
