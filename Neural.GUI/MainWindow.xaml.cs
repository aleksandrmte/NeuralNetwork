using LiveCharts;
using LiveCharts.Wpf;
using Microsoft.Win32;
using Neural.Core;
using Neural.Core.Functions;
using Neural.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;

namespace Neural.GUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private NeuralNetwork _neuralNetwork;
        private double[,] dataSets;
        private double[,] expectedResults;
        private List<double> errors;
        private SeriesCollection seriesCollection;

        public MainWindow()
        {
            InitializeComponent();
            trainButton.Visibility = Visibility.Hidden;
            label1.Visibility = Visibility.Hidden;
            label3.Visibility = Visibility.Hidden;
            label4.Visibility = Visibility.Hidden;
            inputNeuronsCountTextBox.Visibility = Visibility.Hidden;
            dataSetsCountTextBox.Visibility = Visibility.Hidden;
            countEpochTextBox.Visibility = Visibility.Hidden;
        }

        private void ShowElements()
        {
            trainButton.Visibility = Visibility.Visible;
            label1.Visibility = Visibility.Visible;
            label3.Visibility = Visibility.Visible;
            label4.Visibility = Visibility.Visible;
            inputNeuronsCountTextBox.Visibility = Visibility.Visible;
            dataSetsCountTextBox.Visibility = Visibility.Visible;
            countEpochTextBox.Visibility = Visibility.Visible;
        }

        private void button_Click(object sender, RoutedEventArgs e)
        {
            var countOutputs = Convert.ToInt32(countOutputTextBox.Text);
            var hiddenNeuronsCount = Convert.ToInt32(hiddenLayersNeuronsCount.Text);

            var openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                var data = FileReader.Read(openFileDialog.FileName, countOutputs);
                _neuralNetwork = new NeuralNetwork(0.1, new SigmoidFunction(), data.Item1.GetLength(1), hiddenNeuronsCount, countOutputs);

                dataSetsCountTextBox.Text = data.Item1.GetLength(0).ToString();
                inputNeuronsCountTextBox.Text = data.Item1.GetLength(1).ToString();

                dataSets = data.Item1;
                expectedResults = data.Item2;
                ShowElements();
            }

        }

        private void trainButton_Click(object sender, RoutedEventArgs e)
        {
            var epoch = Convert.ToInt32(countEpochTextBox.Text);
            _neuralNetwork.Train(expectedResults, dataSets, epoch);
            errors = _neuralNetwork.Errors;
            MessageBox.Show("Ready");

            seriesCollection = new SeriesCollection
            {
                new LineSeries
                {
                    Values = new ChartValues<double>(errors)
                },

            };
            chart.Series = seriesCollection;
            chart.Update(false, true);
        }
    }
}
