using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class GameManager : MonoBehaviour {
    public GameObject credits;

    private static GameManager m_instance;

    public static GameManager Instance {
        get {
            if (m_instance == null) {
                m_instance = FindObjectOfType<GameManager>();
            }
            return m_instance;
        }
    }

    private void Awake() {
        if (m_instance == null) {
            m_instance = this as GameManager;
        }
    }

    public void StartTraining() {
        NeuralNetworkManager.Instance.Train();
    }

    public void StartGame() {
        SceneManager.LoadScene("One");
    }
}
