using Neural.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using Neural.Core.Functions;
using Neural.Core.Helpers;

namespace Neural.Test
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var network = new NeuralNetwork(0.1, new SigmoidFunction(), 10, 10, 2);
            
            var dataSets = new double[,]
            {
                {1, 1, 1, 0, 1, 0, 1, 0, 1, 0},
                {0, 0, 0, 1, 0, 1, 0, 0, 0, 0},
                {0, 0, 1, 1, 0, 1, 1, 1, 0, 0},
                {1, 1, 0, 0, 1, 0, 0, 1, 1, 0},
                {0, 0, 1, 0, 0, 1, 0, 1, 1, 0},
                {1, 0, 0, 1, 0, 1, 0, 1, 1, 1}
            };
            var expectedResults = new double[] { 1, 0, 0, 1, 0, 0 };
            
            //scaling data [0...1]
            dataSets = network.Scaling(dataSets);

            network.Train(expectedResults, dataSets, 10000);

            //var results = new List<double>();
            //for (var i = 0; i < expectedResults.Length; i++)
            //{
            //    var input = ArrayHelper.GetRow(dataSets, i).ToList();
            //    var result = network.Activate(input);
            //    results.Add(result);
            //}
            //for (var i = 0; i < results.Count; i++)
            //{
            //    var expected = Math.Round(expectedResults[i], 1);
            //    var actual = Math.Round(results[i], 1);
            //    Console.WriteLine($"{expected} = {actual}; {expected == actual}");
            //}

            var inputSignals = new List<double> {1, 1, 0, 0, 1, 0, 1, 0, 1, 0};

            Console.WriteLine(Math.Round(network.Activate(inputSignals), 6));
            network.ShowAll();

            Console.ReadKey();
        }
    }
}
