using System;
using XCharts;
using System.Collections;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using Unity.Jobs;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.SocialPlatforms;
using Debug = UnityEngine.Debug;
using Random = System.Random;

[Serializable]
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

        public TrainingData(MnistReader.Data data, int epoch, int miniBatchSize) {
            var xs = data.xs.Skip(epoch * miniBatchSize).Take(miniBatchSize).ToList();
            var ys = data.ys.Skip(epoch * miniBatchSize).Take(miniBatchSize).ToList();
            for (var i = 0; i < xs.Count; i++) {
                // this[i] = (xs[i], ys[i]);
                List.Add((Vector<double>.Build.DenseOfArray(xs[i]), ys[i]));
            }
        }
    }

    public NeuralNetworkParameters Parameters;

    public int[] sizes;

    private MnistReader.Data testData;
    private MnistReader.Data trainData;

    private static string serializedPath = Application.persistentDataPath + "/neural_network.dat";

    private List<bool> successes;
    private List<Vector<double>> zs;
    private List<Vector<double>> activations;

    private LineChart NNLineChart => NeuralNetworkManager.Instance.lineChart;

    private List<Vector<double>> biases {
        get => Parameters.Biases;
        set => Parameters.Biases = value;
    }

    private List<Matrix<double>> weights {
        get => Parameters.Weights;
        set => Parameters.Weights = value;
    }

    public NeuralNetwork(int[] sizes, MnistReader.Data trainData, MnistReader.Data testData) {
        this.sizes = sizes;
        this.trainData = trainData;
        this.testData = testData;
        successes = new List<bool>();
        activations = new List<Vector<double>>();
        Parameters = new NeuralNetworkParameters(sizes);
        zs = new List<Vector<double>>();
    }

    public void Shuffle() {
        for (var i = 0; i < trainData.xs.Length; i++) {
            var rand = UnityEngine.Random.Range(0, trainData.xs.Length);
            var a = trainData.xs[i];
            var b = trainData.xs[rand];
            var c = trainData.ys[i];
            var d = trainData.ys[rand];
            trainData.xs[i] = b;
            trainData.xs[rand] = a;
            trainData.ys[i] = c;
            trainData.ys[rand] = d;
        }
    }

    public void Load() {
        if (!File.Exists(serializedPath)) {
            return;
        }

        var byteReader = new BinaryReader(File.OpenRead(serializedPath));
        foreach (var i in Enumerable.Range(0, weights.Count)) {
            for (var j = 0; j < sizes[i + 1]; j++) {
                for (var k = 0; k < sizes[i]; k++) {
                    weights[i][j, k] = byteReader.ReadDouble();
                }
            }

            for (var j = 0; j < sizes[i + 1]; j++) {
                biases[i][j] = byteReader.ReadDouble();
            }
        }

        byteReader.Dispose();
    }

    public void Save() {
        var byteWriter = new BinaryWriter(File.OpenWrite(serializedPath));
        foreach (var i in Enumerable.Range(0, weights.Count)) {
            for (var j = 0; j < sizes[i + 1]; j++) {
                for (var k = 0; k < sizes[i]; k++) {
                    byteWriter.Write(BitConverter.GetBytes(weights[i][j, k]));
                }
            }

            for (var j = 0; j < sizes[i + 1]; j++) {
                byteWriter.Write(BitConverter.GetBytes(biases[i][j]));
            }
        }

        byteWriter.Dispose();
    }

    public void SGD(int epochs, int miniBatchSize, double eta) {
        for (var i = 0; i < epochs; i++) {
            var watch = Stopwatch.StartNew();

            // for (var j = 0; j < 10; j += 1) {
            for (var j = 0; j * miniBatchSize < trainData.xs.Length; j += 1) {
                var miniBatch = new TrainingData(trainData, j, miniBatchSize);
                UpdateMiniBatch(miniBatch, eta);
            }

            watch.Stop();
            Debug.LogFormat("Epoch updated time = {0}", watch.ElapsedMilliseconds);
            RunTest();
        }
    }

    public void UpdateMiniBatch(TrainingData trainingData, double eta) {
        Debug.Log("NeuralNetwork: Training!!!");

        var numInEpoch = trainingData.List.Count;

        // NNLineChart.ClearData();
        // NNLineChart.gameObject.SetActive(true);

        // var epochSize = (int) Mathf.Ceil(trainData.Count / epochs);
        //
        // foreach (var epochIdx in Enumerable.Range(0, epochs)) {

        successes = new List<bool>();

        var nablaB = biases.Select(
            b => Vector<double>.Build.Dense(b.Count, 0)
        ).ToList();

        var nablaW = weights.Select(
            w => Matrix<double>.Build.Dense(w.RowCount, w.ColumnCount)
        ).ToList();

        foreach (var (x, y) in trainingData.List) {
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
        weights = weights.Select((w, j) => w - (eta / numInEpoch) * nablaW[j]).ToList();
        biases = biases.Select((b, j) => b - (eta / numInEpoch) * nablaB[j]).ToList();

        // double numerator = successes.FindAll(s => s == true).Count;
        // double denominator = successes.Count;
        // double successRate = numerator / denominator;
        // Debug.LogFormat("epoch finished: successes / total = {0} / {1} -- {2}", numerator, denominator, successRate);

        // NNLineChart.AddData("serie1", epochIdx, (float) successRate);
        // }

        Debug.Log("Done!!");
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

        foreach (var i in Enumerable.Range(0, weights.Count)) {
            var w = weights[i];
            var b = biases[i];
            deltaNablaB.Add(Vector<double>.Build.Dense(b.Count, 0));
            deltaNablaW.Add(Matrix<double>.Build.Dense(w.RowCount, w.ColumnCount, 0));
        }

        foreach (var (b, w) in biases.Zip(weights, Tuple.Create)) {
            z = w.Multiply(activation) + b;
            zs.Add(z);
            activation = z.Map(Sigma);
            activations.Add(activation);
        }

        // Debug.Log("activations:");
        // activations.ForEach(a => Debug.Log(a.ToVectorString()));

        var vecY = ConvertOutputToVector((int) y);
        z = zs.Last();
        var delta = (activation - vecY).PointwiseMultiply(z.Map(SigmaPrime));

        // Debug.Log("delta:");
        // Debug.Log(delta.ToVectorString());

        deltaNablaB[deltaNablaB.Count - 1] = delta;
        deltaNablaW[deltaNablaW.Count - 1] = delta.OuterProduct(activations[activations.Count - 2]);

        for (var l = weights.Count - 2; l >= 0; l--) {
            z = zs[l];
            var sp = z.Map(SigmaPrime);
            delta = weights[l + 1].Transpose().Multiply(delta).PointwiseMultiply(sp);
            deltaNablaB[l] = delta;
            deltaNablaW[l] = delta.OuterProduct(activations[l]);
        }

        // Debug.Log("");
        // Debug.Log("");
        // Debug.Log("");
        // Debug.Log("deltaNablaW:");
        // deltaNablaW.ForEach(dnw => Debug.Log(dnw.ToMatrixString()));
        // Debug.Log("");
        // Debug.Log("");
        // Debug.Log("");
        // Debug.Log("deltaNablaB:");
        // deltaNablaB.ForEach(dnb => Debug.Log(dnb.ToVectorString()));

        return (deltaNablaB, deltaNablaW);
        // nablaB = nablaB.Select((nb, j) => nb + deltaNablaB[j]).ToList();
        // nablaW = nablaW.Select((nw, j) => nw + deltaNablaW[j]).ToList();
    }

    public Vector<double> OutputError(double y) {
        var vectorY = ConvertOutputToVector((int) y);
        var activation = activations.Last();
        var z = zs.Last();
        return (activation - vectorY).PointwiseMultiply(z.Map(SigmaPrime));
    }

    private double CostDerivative(double result, double target) {
        return target - result;
    }

    public Vector<double> Run(byte[] input, byte y) {
        var doubles = input.Select(i => (double) i).ToArray();
        var vec = Vector<double>.Build.Dense(doubles);
        return Run(vec, y);
    }

    public Vector<double> Run(Vector<double> input, byte y) {
        var zs = this.zs;
        var activations = this.activations;

        var activation = input;
        activations = new List<Vector<double>> {input};
        Func<double, double> fn = MathNet.Numerics.SpecialFunctions.Logistic;

        void Fn(Matrix<double> mat, Vector<double> b) {
            var z = mat.Multiply(activation) + b;
            zs.Add(z);
            activation = z.Map(fn);
            activations.Add(activation);
        }

        foreach (var i in Enumerable.Range(0, weights.Count)) {
            var mat = weights[i];
            var b = biases[i];
            Fn(mat, b);
        }

        var result = activations.Last();

        successes.Add(result.MaximumIndex() == y);

        return result;
    }

    public void RunTest() {
        var successes = 0;
        var data = new TrainingData(testData, 0, testData.xs.Length);

        foreach (var (x, y) in data.List) {
            Vector<double> z;
            var activation = x;

            foreach (var (b, w) in biases.Zip(weights, Tuple.Create)) {
                z = w.Multiply(activation) + b;
                zs.Add(z);
                activation = z.Map(Sigma);
                activations.Add(activation);
            }

            var result = activation.MaximumIndex();

            if (result == y) {
                successes++;
            }
        }


        Debug.LogFormat("total successes {0}, total {1}, percentage {2}", successes, data.List.Count,
            (double) successes / (double) data.List.Count);
    }

    private void Init() {
        Debug.Log("Init!!");
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