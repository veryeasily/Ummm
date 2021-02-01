using Ummm;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using Debug = UnityEngine.Debug;
using Unity.Barracuda;

public class NeuralNetwork {
    public class NeuralNetworkParameters {
        public List<Vector<double>> Biases = new List<Vector<double>>();
        public List<Matrix<double>> Weights = new List<Matrix<double>>();

        public NeuralNetworkParameters(IReadOnlyList<int> sizes) {
            for (var i = 1; i < sizes.Count; i++) {
                var size = sizes[i];
                var prevSize = sizes[i - 1];
                Biases.Add(Vector<double>.Build.Random(size, Normal.WithMeanVariance(0, 0.5d)));
                Weights.Add(Matrix<double>.Build.Random(size, prevSize, Normal.WithMeanVariance(0, 0.5d)));
            }
        }
    }

    public class TrainingData {
        public readonly List<(Vector<double>, double)> List = new List<(Vector<double>, double)>();

        public TrainingData(Dataset data, int epoch, int miniBatchSize) {
            var xs = data.Xs.Skip(epoch * miniBatchSize).Take(miniBatchSize).ToList();
            var ys = data.Ys.Skip(epoch * miniBatchSize).Take(miniBatchSize).ToList();
            for (var i = 0; i < xs.Count; i++) {
                // this[i] = (xs[i], ys[i]);
                List.Add((Vector<double>.Build.DenseOfArray(xs[i]), ys[i]));
            }
        }
    }

    public NeuralNetworkParameters Parameters;

    public int[] Sizes;

    private Dataset _testData;
    private Dataset _trainData;

    private CancellationTokenSource _source = new CancellationTokenSource();

    private static string _serializedPath = UnityEngine.Application.persistentDataPath + "/neural_network.dat";

    private List<bool> _successes;
    private List<Vector<double>> _zs;
    private List<Vector<double>> _activations;

    private int _epochs;
    private int _miniBatchSize;
    private double _eta;
    private Task task;

    private List<Vector<double>> Biases {
        get => Parameters.Biases;
        set => Parameters.Biases = value;
    }

    private List<Matrix<double>> Weights {
        get => Parameters.Weights;
        set => Parameters.Weights = value;
    }

    public NeuralNetwork(int[] sizes, Dataset trainData, Dataset testData) {
        this.Sizes = sizes;
        this._trainData = trainData;
        this._testData = testData;
        _successes = new List<bool>();
        _activations = new List<Vector<double>>();
        Parameters = new NeuralNetworkParameters(sizes);
        _zs = new List<Vector<double>>();
    }

    public void Load() {
        if (!File.Exists(_serializedPath)) {
            return;
        }

        var byteReader = new BinaryReader(File.OpenRead(_serializedPath));
        foreach (var i in Enumerable.Range(0, Weights.Count)) {
            for (var j = 0; j < Sizes[i + 1]; j++) {
                for (var k = 0; k < Sizes[i]; k++) {
                    Weights[i][j, k] = byteReader.ReadDouble();
                }
            }

            for (var j = 0; j < Sizes[i + 1]; j++) {
                Biases[i][j] = byteReader.ReadDouble();
            }
        }

        byteReader.Dispose();
    }

    public void Save() {
        var byteWriter = new BinaryWriter(File.OpenWrite(_serializedPath));
        foreach (var i in Enumerable.Range(0, Weights.Count)) {
            for (var j = 0; j < Sizes[i + 1]; j++) {
                for (var k = 0; k < Sizes[i]; k++) {
                    byteWriter.Write(BitConverter.GetBytes(Weights[i][j, k]));
                }
            }

            for (var j = 0; j < Sizes[i + 1]; j++) {
                byteWriter.Write(BitConverter.GetBytes(Biases[i][j]));
            }
        }

        byteWriter.Dispose();
    }

    public void Sgd(int epochs, int miniBatchSize, double eta) {
        this._epochs = epochs;
        this._miniBatchSize = miniBatchSize;
        this._eta = eta;
        var token = _source.Token;

        task = Task.Factory.StartNew(() =>
        {
            Debug.Log("Start of SGD on Thread!");

            for (var i = 0; i < epochs; i++) {
                if (token.IsCancellationRequested) {
                    Debug.LogFormat("Early returning from start of epoch {0}...", i);
                    return;
                }

                Debug.LogFormat("Starting epoch {0}", i);
                var watch = Stopwatch.StartNew();

                _trainData.Shuffle();

                Debug.LogFormat("Past shuffle...");

                // for (var j = 0; j < 10; j += 1) {
                for (var j = 0; j * miniBatchSize < _trainData.Xs.Length; j += 1) {
                    if (token.IsCancellationRequested) {
                        Debug.LogFormat("Early returning from start of mini-batch {0}...", j);
                        return;
                    }

                    // var miniBatch = new TrainingData(_trainData, j, miniBatchSize);
                    var miniBatch = _trainData.GetMiniBatch(j, miniBatchSize);
                    UpdateMiniBatch(miniBatch, eta);
                }

                watch.Stop();
                Debug.LogFormat("Epoch updated time = {0}", watch.ElapsedMilliseconds);
                RunTest();
            }

            Debug.Log("End of SGD on Thread!");

            if (token.IsCancellationRequested) {
                Debug.LogFormat("Early returning before Save() call...");
                return;
            }

            Save();

            Debug.Log("Done saving!");
        }, token);
    }

    public void Cancel() {
        Debug.Log("NeuralNetwork#Cancel");
        _source?.Cancel();
    }

    public Vector<double> OutputError(double y) {
        var vectorY = ConvertOutputToVector((int) y);
        var activation = _activations.Last();
        var z = _zs.Last();
        return (activation - vectorY).PointwiseMultiply(z.Map(SigmaPrime));
    }

    public Vector<double> Run(byte[] input, byte y) {
        var doubles = input.Select(i => (double) i).ToArray();
        var vec = Vector<double>.Build.Dense(doubles);
        return Run(vec, y);
    }

    private void UpdateMiniBatch(List<(Vector<double>, double)> miniBatch, double eta) {
        var numInEpoch = miniBatch.Count;

        // NNLineChart.ClearData();
        // NNLineChart.gameObject.SetActive(true);

        // var epochSize = (int) Mathf.Ceil(trainData.Count / epochs);
        //
        // foreach (var epochIdx in Enumerable.Range(0, epochs)) {

        _successes = new List<bool>();

        var nablaB = Biases.Select(
            b => Vector<double>.Build.Dense(b.Count, 0)
        ).ToList();

        var nablaW = Weights.Select(
            w => Matrix<double>.Build.Dense(w.RowCount, w.ColumnCount)
        ).ToList();

        foreach (var (x, y) in miniBatch) {
            var (deltaNablaB, deltaNablaW) = Backprop(x, y);
            for (var k = 0; k < nablaB.Count; k++) {
                nablaB[k] = nablaB[k] + deltaNablaB[k];
                nablaW[k] = nablaW[k] + deltaNablaW[k];
            }
        }

        // foreach (var itemIdx in Enumerable.Range(0, numInEpoch)) {
        //     var (x, y) = trainingData[itemIdx];
        // }

        // for (var i = 0; i < weights.Count; i++) {
        //     var w = weights[i];
        //     weights[i] = w - (eta / numInEpoch) * nablaW[i];
        // }
        Weights = Weights.Select((w, j) => w - (eta / numInEpoch) * nablaW[j]).ToList();
        Biases = Biases.Select((b, j) => b - (eta / numInEpoch) * nablaB[j]).ToList();

        // double numerator = successes.FindAll(s => s == true).Count;
        // double denominator = successes.Count;
        // double successRate = numerator / denominator;
        // Debug.LogFormat("epoch finished: successes / total = {0} / {1} -- {2}", numerator, denominator, successRate);

        // NNLineChart.AddData("serie1", epochIdx, (float) successRate);
        // }
    }

    private (List<Vector<double>>, List<Matrix<double>>) Backprop(Vector<double> x, double y) {
        // nabla_b = [np.zeros(b.shape) for b in self.biases]
        // nabla_w = [np.zeros(w.shape) for w in self.weights]
        // # feedforward
        // activation = x
        // activations = [x] # list to store all the activations, layer by layer
        // zs = [] # list to store all the z vectors, layer by layer
        // for b, w in zip(self.biases, self.weights):
        //     z = np.dot(w, activation)+b
        //     zs.append(z)
        //     activation = sigmoid(z)
        //     activations.append(activation)

        var deltaNablaB = new List<Vector<double>>();
        var deltaNablaW = new List<Matrix<double>>();
        var activation = x;
        var activations = new List<Vector<double>>() {x};
        var zs = new List<Vector<double>>();
        Vector<double> z;

        foreach (var i in Enumerable.Range(0, Weights.Count)) {
            var w = Weights[i];
            var b = Biases[i];
            deltaNablaB.Add(Vector<double>.Build.Dense(b.Count, 0));
            deltaNablaW.Add(Matrix<double>.Build.Dense(w.RowCount, w.ColumnCount, 0));
        }

        foreach (var (b, w) in Biases.Zip(Weights, Tuple.Create)) {
            z = w.Multiply(activation) + b;
            zs.Add(z);
            activation = z.Map(Sigma);
            activations.Add(activation);
        }

        var vecY = ConvertOutputToVector((int) y);
        z = zs.Last();
        var delta = (activation - vecY).PointwiseMultiply(z.Map(SigmaPrime));

        deltaNablaB[deltaNablaB.Count - 1] = delta;
        deltaNablaW[deltaNablaW.Count - 1] = delta.OuterProduct(activations[activations.Count - 2]);

        for (var l = Weights.Count - 2; l >= 0; l--) {
            z = zs[l];
            var sp = z.Map(SigmaPrime);
            delta = Weights[l + 1].Transpose().Multiply(delta).PointwiseMultiply(sp);
            deltaNablaB[l] = delta;
            deltaNablaW[l] = delta.OuterProduct(activations[l]);
        }

        return (deltaNablaB, deltaNablaW);
    }

    private double CostDerivative(double result, double target) {
        return target - result;
    }

    private Vector<double> Run(Vector<double> input, byte y) {
        var zs = this._zs;
        var activations = this._activations;

        var activation = input;
        activations = new List<Vector<double>> {input};
        Func<double, double> fn = MathNet.Numerics.SpecialFunctions.Logistic;

        void Fn(Matrix<double> mat, Vector<double> b) {
            var z = mat.Multiply(activation) + b;
            zs.Add(z);
            activation = z.Map(fn);
            activations.Add(activation);
        }

        foreach (var i in Enumerable.Range(0, Weights.Count)) {
            var mat = Weights[i];
            var b = Biases[i];
            Fn(mat, b);
        }

        var result = activations.Last();

        _successes.Add(result.MaximumIndex() == y);

        return result;
    }

    public void RunTest() {
        var successes = 0;
        var data = new TrainingData(_testData, 0, _testData.Xs.Length);

        foreach (var (x, y) in data.List) {
            Vector<double> z;
            var activation = x;

            foreach (var (b, w) in Biases.Zip(Weights, Tuple.Create)) {
                z = w.Multiply(activation) + b;
                _zs.Add(z);
                activation = z.Map(Sigma);
                _activations.Add(activation);
            }

            var result = activation.MaximumIndex();

            if (result == y) {
                successes++;
            }
        }


        Debug.LogFormat("total successes {0}, total {1}, percentage {2}", successes, data.List.Count,
            (double) successes / (double) data.List.Count);
    }

    private Vector<double> ConvertOutputToVector(int y) {
        var vector = Vector<double>.Build.Dense(10, 0);
        vector[y] = 1f;
        return vector;
    }

    private double Sigma(double z) {
        return (double) MathNet.Numerics.SpecialFunctions.Logistic(z);
    }

    private double SigmaPrime(double z) {
        return Sigma(z) * (1d - Sigma(z));
    }
}