using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;

public class PlayerController : MonoBehaviour
{
    public float speed = 1f;

    private Rigidbody _rb;
    private Vector2 _moveVec;

    void Start()
    {
        _rb = GetComponent<Rigidbody>();
    }

    private void OnTriggerEnter(Collider other) {
        if (other.CompareTag("Touchable")) {
            Debug.Log("Beginning neural network computation...");
            other.gameObject.SetActive(false);
            GameManager.Instance.StartTraining();
        }
    }

    private void OnMove(InputValue movementValue) {
        _moveVec = movementValue.Get<Vector2>();
    }

    private void OnQuit(InputValue keyValue) {
        Application.Quit();
    }

    private void OnNextLevel(InputValue value) {
        Debug.Log(value.ToString());
        Debug.Log(value);
        SceneManager.LoadScene("Two", LoadSceneMode.Single);
    }

    void FixedUpdate()
    {
        Vector3 updated = new Vector3(_moveVec.x, 0f, _moveVec.y);
        _rb.AddForce(updated * speed);
    }
}
