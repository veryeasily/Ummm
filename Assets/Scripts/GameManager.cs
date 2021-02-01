using System.Collections;
using System.Collections.Generic;
using Ummm;
using UnityEngine;
using UnityEngine.SceneManagement;

public class GameManager : MonoBehaviour {
    public GameObject credits;

    private static GameManager _mInstance;

    public static GameManager Instance {
        get {
            if (_mInstance == null) {
                _mInstance = FindObjectOfType<GameManager>();
            }
            return _mInstance;
        }
    }

    private void Awake() {
        if (_mInstance == null) {
            _mInstance = this as GameManager;
        }
    }

    public void StartTraining() {
        NeuralNetworkManager.Instance.Train();
    }

    public void StartGame() {
        SceneManager.LoadScene("One");
    }
}
