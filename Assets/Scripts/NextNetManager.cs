using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

namespace Ummm {
    public class NextNetManager : MonoBehaviour {
        private static NextNetManager _mInstance;
        
        public NNModel modelSource;
        
        public static NextNetManager Instance {
            get {
                if (_mInstance == null) {
                    _mInstance = FindObjectOfType<NextNetManager>();
                }

                return _mInstance;
            }
        }

        private void Awake() {
            if (_mInstance == null) {
                _mInstance = this as NextNetManager;
            }
        }

        // Start is called before the first frame update
        private void Start()
        {
            var model = ModelLoader.Load(modelSource);
            var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

            var inputTensor = new Tensor(1, 2, new float[2] { 0, 0 });
            worker.Execute(inputTensor);

            var output = worker.PeekOutput();
            print("This is the output: " + (output[0] < 0.5? 0 : 1));
        
            inputTensor.Dispose();
            output.Dispose();
            worker.Dispose();
        }
    }
}