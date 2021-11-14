using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerControllerTwo : MonoBehaviour
{
    public void OnQuit(InputValue input) {
        Debug.LogFormat("InputValue = {0}", input);
        Application.Quit();
    }
}
