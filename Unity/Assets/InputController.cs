using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputController : MonoBehaviour
{
    private DuckieControl charController;

    void Awake()
    {
        charController = GetComponent<DuckieControl>();
    }

    private void Update()
    {
        // Get input values
        int vertical = Mathf.RoundToInt(Input.GetAxis("Vertical"));
        int horizontal = Mathf.RoundToInt(Input.GetAxis("Horizontal"));

        charController.ForwardInput = vertical;
        charController.TurnInput = horizontal;
    }
}
