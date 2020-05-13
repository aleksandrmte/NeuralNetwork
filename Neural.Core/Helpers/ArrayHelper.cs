﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Neural.Core.Helpers
{
    public class ArrayHelper
    {
        public static T[] GetColumn<T>(T[,] matrix, int columnNumber)
        {
            return Enumerable.Range(0, matrix.GetLength(0))
                .Select(x => matrix[x, columnNumber])
                .ToArray();
        }

        public static T[] GetRow<T>(T[,] matrix, int rowNumber)
        {            
            return Enumerable.Range(0, matrix.GetLength(1))
                .Select(x => matrix[rowNumber, x])
                .ToArray();
        }

        public static IEnumerable<T> Each<T>(T[,] source)
        {
            return source.Cast<T>();
        }
    }
}
