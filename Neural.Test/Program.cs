using Neural.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using Neural.Core.Functions;
using Neural.Core.Helpers;
using Neural.IO;
using Neural.Core.Models;

namespace Neural.Test
{
    internal class Program
    {
        private const int CountEpoch = 50000;

        private static void Main(string[] args)
        {
            Console.WriteLine("App running");

            const int countOutputs = 4;

            var data = FileReader.Read(@"D:\auto.csv", countOutputs);
            Console.WriteLine("Data from file loaded.");

            var network = new NeuralNetwork(0.01, new SigmoidFunction(), data.Input.GetLength(1), 4, countOutputs);
            Console.WriteLine("Neural network created.");

            var dataSets = data.Input;
            var expectedResults = data.Output;

            
            if (ArrayHelper.Each(dataSets).Any(x => x > 1 || x < 0))
            {
                //scaling data [0...1]
                dataSets = network.Scaling(dataSets);
                Console.WriteLine("Data scaled.");
            }

            Console.WriteLine("Start train...");
            network.Train(expectedResults, dataSets, CountEpoch);
            Console.WriteLine("End train.");

            //Compare data after training
            var results = new List<List<double>>();
            for (var i = 0; i < expectedResults.GetLength(0); i++)
            {
                var input = ArrayHelper.GetRow(dataSets, i).ToList();
                var result = network.Activate(input);
                results.Add(result);
            }
            Console.WriteLine("Results:");
            for (var i = 0; i < results.Count; i++)
            {
                for (var j = 0; j < results[i].Count; j++)
                {
                    var expected = Math.Round(expectedResults[i, j]);
                    var actual = Math.Round(results[i][j], 1);
                    Console.WriteLine($"{expected} = {actual}; {expected == actual}");
                }
                Console.WriteLine();
            }
            Console.ReadKey();
        }
    }
}
