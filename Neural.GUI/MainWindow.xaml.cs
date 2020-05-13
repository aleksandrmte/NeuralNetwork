using LiveCharts;
using LiveCharts.Wpf;
using Microsoft.Win32;
using Neural.Core;
using Neural.Core.Functions;
using Neural.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Windows;
using System.Windows.Threading;
using Neural.Core.Helpers;

namespace Neural.GUI
{
    public partial class MainWindow : Window
    {
        private NeuralNetwork _neuralNetwork;
        private double[,] _dataSets;
        private double[,] _expectedResults;
        private List<double> _errors;
        private SeriesCollection _seriesCollection;
        private int _epoch;
        private Thread _thread;
        private string _statusText = "";
        private DispatcherTimer _dispatcherTimer;
        private DateTime _timeStart;
        private bool _isProcessed;

        public MainWindow()
        {
            InitializeComponent();
            HideElements();
        }

        private void HideElements()
        {
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
            CreateTimer();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() != true)
                return;

            var countOutputs = Convert.ToInt32(countOutputTextBox.Text);
            var hiddenNeuronsCount = Convert.ToInt32(hiddenLayersNeuronsCount.Text);

            var data = FileReader.Read(openFileDialog.FileName, countOutputs);
            _neuralNetwork = new NeuralNetwork(0.1, new SigmoidFunction(), data.Input.GetLength(1), hiddenNeuronsCount, countOutputs);

            dataSetsCountTextBox.Text = data.Input.GetLength(0).ToString();
            inputNeuronsCountTextBox.Text = data.Input.GetLength(1).ToString();

            _dataSets = data.Input;
            Scaling();

            _expectedResults = data.Output;
            ShowElements();
        }

        private void CreateTimer()
        {
            _dispatcherTimer = new DispatcherTimer();
            _dispatcherTimer.Tick += DispatcherTimer_Tick;
            _dispatcherTimer.Interval = new TimeSpan(0, 0, 0, 0, 200);
        }

        private void DispatcherTimer_Tick(object sender, EventArgs e)
        {
            var time = _isProcessed? " (" + (DateTime.Now - _timeStart).Seconds + ")": "";
            lblStatus.Text = _statusText + time;
        }

        private void TrainButton_Click(object sender, RoutedEventArgs e)
        {
            if (!_dispatcherTimer.IsEnabled)
                _dispatcherTimer.Start();
            _epoch = Convert.ToInt32(countEpochTextBox.Text);
            _statusText = "Computing...";
            _timeStart = DateTime.Now;
            _thread = new Thread(Training);
            _thread.Start();
        }

        private void Training()
        {
            _isProcessed = true;
            _neuralNetwork.Train(_expectedResults, _dataSets, _epoch);
            _errors = _neuralNetwork.Errors;
            _statusText = "Computed. Drawing chart...";
            Thread.Sleep(500);
            DrawChart();
        }

        private void Scaling()
        {
            if (ArrayHelper.Each(_dataSets).Any(x => x > 1 || x < 0))
            {
                _dataSets = _neuralNetwork.Scaling(_dataSets);
            }
        }

        private void DrawChart()
        {
            var groupedData = GroupData(_errors);
            _statusText = "Finished";
            _isProcessed = false;

            Dispatcher.Invoke(() =>
            {
                _seriesCollection = new SeriesCollection
                {
                    new LineSeries
                    {
                        Values = new ChartValues<double>(groupedData)
                    }
                };
                chart.Series = _seriesCollection;
                chart.Update(true, true);
            });
        }

        private static IEnumerable<double> GroupData(IReadOnlyCollection<double> data)
        {
            const decimal countPoints = 1000;

            if (data.Count <= countPoints)
                return data;

            var groupedData = new List<double>();

            var count = (int)Math.Ceiling(data.Count / countPoints);

            for (var i = 0; i < countPoints; i++)
            {
                var point = data.Skip(i * count).Take(count).Average();
                groupedData.Add(point);
            }

            return groupedData;
        }
    }
}
