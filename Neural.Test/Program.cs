using Neural.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using Neural.Core.Functions;
using Neural.Core.Helpers;
using System.IO;
using System.Globalization;

namespace Neural.Test
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var data = ReadFile();
            var k = data.Item1.GetLength(1);
            var network = new NeuralNetwork(0.1, new SigmoidFunction(), data.Item1.GetLength(1), data.Item1.GetLength(1), 1);

            //var network = new NeuralNetwork(0.1, new SigmoidFunction(), 4, 4, 1);
            //var dataSets = new double[,]
            //{
            //    {0,1,0,0},
            //    {1,0,0,0},
            //    { 0,1,1,1},
            //    {1,0,1,0 },
            //    {1,1,1,1 },
            //    {0,1,0,0 }
            //};
            //var expectedResults = new double[] { 1, 0, 0, 0, 1, 1 };


            var dataSets = data.Item1;
            var expectedResults = data.Item2;

            //scaling data [0...1]
            dataSets = network.Scaling(dataSets);

            network.Train(expectedResults, dataSets, 20000);

            var results = new List<double>();
            for (var i = 0; i < expectedResults.Length; i++)
            {
                var input = ArrayHelper.GetRow(dataSets, i).ToList();
                var result = network.Activate(input);
                results.Add(result);
            }
            for (var i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(expectedResults[i], 1);
                var actual = Math.Round(results[i], 1);
                Console.WriteLine($"{expected} = {actual}; {expected == actual}");
            }

            //var inputSignals = new List<double> { 1, 1, 1, 1 };

            //Console.WriteLine(Math.Round(network.Activate(inputSignals), 6));
            //network.ShowAll();

            Console.ReadKey();
        }

        private static (double[,], double[]) ReadFile()
        {            
            var content = File.ReadAllLines(@"G:\arend\heart-disease-uci\heart.csv");

            var rowCount = content.Count();
            var columnCount = content[0].Split(";").Count() - 1;
            var dataSets = new double[rowCount, columnCount];
            var expectedResults = new double[rowCount];

            var i = 0;
            foreach (var row in content.Skip(1))
            {
                
                var rowValues = row.Split(";");

                for (var j = 0; j < rowValues.Length -1; j++)
                {
                    dataSets[i, j] = Convert.ToDouble(rowValues[j], CultureInfo.InvariantCulture);
                }

                expectedResults[i] = Convert.ToDouble(rowValues[rowValues.Length - 1]);
                i++;
            }

            return (dataSets, expectedResults);
        }
    }
}
