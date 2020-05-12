using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace Neural.IO
{
    public class FileReader
    {
        public static (double[,], double[,]) Read(string path, int countOutputs)
        {
            var content = File.ReadAllLines(path);
            var rowCount = content.Length - 1;
            var columnCount = content[0].Split(";").Length - 1;
            var dataSets = new double[rowCount, columnCount];
            var expectedResults = new double[rowCount, countOutputs];
            var i = 0;
            foreach (var row in content.Skip(1))
            {
                var rowValues = row.Split(";");
                for (var j = 0; j < rowValues.Length - 1; j++)
                {
                    dataSets[i, j] = Convert.ToDouble(rowValues[j], CultureInfo.InvariantCulture);
                }

                var results = rowValues[^1].Split("-");
                for (var k = 0; k < countOutputs; k++)
                {
                    expectedResults[i, k] = Convert.ToDouble(results[k]);
                }
                
                i++;
            }
            return (dataSets, expectedResults);
        }
    }
}
