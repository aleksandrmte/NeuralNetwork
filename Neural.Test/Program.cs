using Neural.Core;
using System;
using System.Collections.Generic;
using Neural.Core.Functions;

namespace Neural.Test
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var network = new NeuralNetwork(0.1, new SigmoidFunction(), 4, 2, 1);
            
            //Вы любите горы?
            //Вы любите путешествовать?
            //Вы любите города?
            //Вы любите играть на компьютере?
            //Вы предпочитаете активный отдых?
            //Вы любите сидеть дома?

            var expectedResults = new List<double> {  1, 0, 0, 1, 0, 0 };


            var dataSets = new List<List<double>>
            {
                new List<double> {1, 1, 1, 0, 1, 0},
                new List<double> {0, 0, 0, 1, 0, 1},
                new List<double> {0, 0, 1, 1, 0, 1},
                new List<double> {1, 1, 0, 0, 1, 0},
                new List<double> {0, 0, 1, 0, 0, 1},
                new List<double> {1, 0, 0, 1, 0, 1}
            };

            network.SetExpectedResults(expectedResults);
            network.Train(10000, dataSets);

            //for (var k = 0; k < 20; k++)
            //{
            //    var results = new List<double>();
            //    for (var i = 0; i < expectedResults.Count; i++)
            //    {

            //        var result = network.Activate(dataSets[i]);
            //        results.Add(result);
            //    }
            //    for (var i = 0; i < results.Count; i++)
            //    {
            //        var expected = Math.Round(expectedResults[i], 1);
            //        var actual = Math.Round(results[i], 1);
            //        Console.WriteLine($"{expected} - {actual}; {expected == actual}");

            //    }
            //    Console.WriteLine();
            //}

            var inputSignals = new List<double> { 1, 1, 0, 0, 1, 1 };

            Console.WriteLine(Math.Round(network.Activate(inputSignals), 6));

            Console.ReadKey();
        }
    }
}
