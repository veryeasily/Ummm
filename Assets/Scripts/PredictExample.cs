using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;

public class PredictExample : MonoBehaviour
{
    public NNModel modelSource;

    void Awake() {
        Debug.Log("test... test...");
    }

    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("At beginning of Start...");
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
