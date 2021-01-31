using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerController : MonoBehaviour
{
    public float speed = 1f;

    private Rigidbody rb;
    private Vector2 moveVec;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    private void OnTriggerEnter(Collider other) {
        if (other.CompareTag("Touchable")) {
            Debug.Log("Beginning neural network computation...");
            other.gameObject.SetActive(false);
            GameManager.Instance.StartTraining();
        }
    }

    private void OnMove(InputValue movementValue) {
        moveVec = movementValue.Get<Vector2>();
    }

    private void OnQuit(InputValue keyValue) {
        Application.Quit();
    }

    void FixedUpdate()
    {
        Vector3 updated = new Vector3(moveVec.x, 0f, moveVec.y);
        rb.AddForce(updated * speed);
    }
}
