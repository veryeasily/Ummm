using UnityEngine;

namespace Ummm {
    public class NeuralNetworkManager : MonoBehaviour {
        public int epochs = 30;
        public int miniBatchSize = 10;
        public double learningRate = 3.0f;
        public int[] sizes = {784, 30, 10};

        private NeuralNetwork _nn;
        private static NeuralNetworkManager _mInstance;

        [System.Serializable]
        public struct DatasetAssets {
            public TextAsset images;
            public TextAsset labels;

            public Dataset Read() {
                return Dataset.ReadAssets(images.bytes, labels.bytes);
            }
        }

        public DatasetAssets trainDatasetAssets;
        public DatasetAssets testDatasetAssets;

        public static NeuralNetworkManager Instance {
            get
            {
                if (_mInstance == null) {
                    _mInstance = FindObjectOfType<NeuralNetworkManager>();
                }

                return _mInstance;
            }
        }

        public void Train() {
            var testData = Dataset.ReadAssets(
                testDatasetAssets.images.bytes,
                testDatasetAssets.labels.bytes
            );
            var trainData = Dataset.ReadAssets(
                trainDatasetAssets.images.bytes,
                trainDatasetAssets.labels.bytes
            );
            _nn = new NeuralNetwork(sizes, trainData, testData);
            _nn.Load();
            _nn.Sgd(epochs, miniBatchSize, learningRate);
        }

        private void Awake() {
            if (_mInstance == null) {
                _mInstance = this as NeuralNetworkManager;
            }
        }

        private void OnDestroy() {
            Debug.Log("NeuralNetworkManager#OnDisable");
            _nn?.Cancel();
        }
    }
}