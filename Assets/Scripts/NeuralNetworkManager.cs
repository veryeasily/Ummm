using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using UnityEngine;
using XCharts;

public class NeuralNetworkManager : MonoBehaviour {
    public int Epochs = 30;
    public int MiniBatchSize = 10;
    public LineChart lineChart;
    public double LearningRate = 3.0f;
    public int[] sizes = {784, 30, 10};

    private NeuralNetwork neuralNetwork;
    private static NeuralNetworkManager m_instance;

    [Serializable]
    public struct Dataset {
        public TextAsset images;
        public TextAsset labels;
    }

    public Dataset TrainDataset;
    public Dataset TestDataset;

    public static NeuralNetworkManager Instance {
        get
        {
            if (m_instance == null) {
                m_instance = FindObjectOfType<NeuralNetworkManager>();
            }

            return m_instance;
        }
    }

    public void Train() {
        var testData = MnistReader.ReadAssets(
            TestDataset.images,
            TestDataset.labels
        );
        var trainData = MnistReader.ReadAssets(
            TrainDataset.images,
            TrainDataset.labels
        );
        // // var neuralNetwork = new NeuralNetwork(sizes, trainData, testData 60, 3f, sizes, testData, trainData);
        var neuralNetwork = new NeuralNetwork(sizes, trainData, testData);
        neuralNetwork.Load();
        neuralNetwork.SGD(Epochs, MiniBatchSize, LearningRate);
        neuralNetwork.Save();
    }

    private void Awake() {
        if (m_instance == null) {
            m_instance = this as NeuralNetworkManager;
        }
    }
}