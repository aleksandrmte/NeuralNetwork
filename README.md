# Simple neural network

Usage:
```
var network = new NeuralNetwork(0.7, new SigmoidFunction(), 2, 2, 1);
var dataSet = new List<double> { 1, 1 };
var result = network.Activate(dataSet);
Console.WriteLine(Math.Round(result[0], 2));
```

Train network:
```
var dataSets = new double[4,2];
dataSets[0, 0] = 0; dataSets[0, 1] = 0;
dataSets[1, 0] = 0; dataSets[1, 1] = 1;
dataSets[2, 0] = 1; dataSets[2, 1] = 0;
dataSets[3, 0] = 1; dataSets[3, 1] = 1;

var expectedResults = new double[4,1];
expectedResults[0, 0] = 0;
expectedResults[1, 0] = 0;
expectedResults[2, 0] = 0;
expectedResults[3, 0] = 1;

var countEpoch = 5000;
var learningRate = 0.7;
var layersNeuronsCount = new[] { 2, 2, 1 };
var network = new NeuralNetwork(learningRate, new SigmoidFunction(), layersNeuronsCount);
network.Train(expectedResults, dataSets, countEpoch);
```

![Result](https://github.com/aleksandrmte/NeuralNetwork/blob/master/DataSets/example.jpg)
