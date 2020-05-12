using System;
using System.Collections.Generic;
using System.Text;

namespace Neural.Core.Helpers
{
    public class RandomHelper
    {
        private static readonly Random Random = new Random();
        public static double GetRandom()
        {
            return 2 * Random.NextDouble() - 1;
        }
    }
}
