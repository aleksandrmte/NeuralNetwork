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
        public static (double[,], double[,]) Read(string path)
        {
            var content = File.ReadAllLines(path);
            var rowCount = content.Length;
            var columnCount = content[0].Split(";").Length - 1;
            var dataSets = new double[rowCount, columnCount];
            var expectedResults = new double[rowCount, 1];
            var i = 0;
            foreach (var row in content.Skip(1))
            {
                var rowValues = row.Split(";");
                for (var j = 0; j < rowValues.Length - 1; j++)
                {
                    dataSets[i, j] = Convert.ToDouble(rowValues[j], CultureInfo.InvariantCulture);
                }
                expectedResults[i, 0] = Convert.ToDouble(rowValues[^1]);
                i++;
            }
            return (dataSets, expectedResults);
        }
    }
}
