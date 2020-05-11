# Simple neural network

Usage:
```
var network = new NeuralNetwork(0.1, new SigmoidFunction(), 4, 2, 1);
var inputSignals = new List<double> { 1, 1, 0, 0, 1, 1 };
var result = Math.Round(network.Activate(inputSignals), 3);
```

Train network:
```
var network = new NeuralNetwork(0.1, new SigmoidFunction(), 4, 2, 1);
var dataSets = new List<List<double>>
            {
                new List<double> {1, 1, 1, 0, 1, 0},
                new List<double> {0, 0, 0, 1, 0, 1},
                new List<double> {0, 0, 1, 1, 0, 1},
                new List<double> {1, 1, 0, 0, 1, 0},
                new List<double> {0, 0, 1, 0, 0, 1},
                new List<double> {1, 0, 0, 1, 0, 1}
            };
var expectedResults = new List<double> {  1, 0, 0, 1, 0, 0 };
network.SetExpectedResults(expectedResults);
network.Train(10000, dataSets);
```

