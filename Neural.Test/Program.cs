﻿using Neural.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using Neural.Core.Functions;
using Neural.Core.Helpers;
using Neural.IO;

namespace Neural.Test
{
    internal class Program
    {
        private const int CountEpoch = 10000;

        private static void Main(string[] args)
        {
            Console.WriteLine("App running");
            
            var data = FileReader.Read(@"D:\heart.csv");
            Console.WriteLine("Data from file loaded.");

            var network = new NeuralNetwork(0.1, new SigmoidFunction(), data.Item1.GetLength(1), 10, 1);
            Console.WriteLine("Neural network created.");
            
            var dataSets = data.Item1;
            var expectedResults = data.Item2;

            //scaling data [0...1]
            dataSets = network.Scaling(dataSets);
            Console.WriteLine("Data scaled.");

            Console.WriteLine("Start train...");
            network.Train(expectedResults, dataSets, CountEpoch);
            Console.WriteLine("End train.");

            //Compare data after training
            var results = new List<double>();
            for (var i = 0; i < expectedResults.Length; i++)
            {
                var input = ArrayHelper.GetRow(dataSets, i).ToList();
                var result = network.Activate(input);
                results.Add(result[0]);
            }
            for (var i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(expectedResults[i,0], 1);
                var actual = Math.Round(results[i], 1);
                Console.WriteLine($"{expected} = {actual}; {expected == actual}");
            }

            Console.ReadKey();
        }
    }
}
