using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DuckieControl : MonoBehaviour
{
    [Tooltip("Move speed in meters/second")]
    public float moveSpeed = 0.1f;
    [Tooltip("Turn speed in degrees/second, left (+) or right (-)")]
    public float turnSpeed = 30;
    public float ForwardInput { get; set; }
    public float TurnInput { get; set; }
    new private Rigidbody rigidbody;

    private void Awake()
    {
        rigidbody = GetComponent<Rigidbody>();
    }
    private void FixedUpdate()
    {
        ProcessActions();
    }
    
    private void ProcessActions()
    {
        // Turning
        if (TurnInput != 0f)
        {
            float angle = Mathf.Clamp(TurnInput, -1f, 1f) * turnSpeed;
            transform.Rotate(Vector3.up, Time.fixedDeltaTime * angle);
        }

        // Movement
        Vector3 move = transform.forward * Mathf.Clamp(ForwardInput, -1f, 1f) *
                       moveSpeed * Time.fixedDeltaTime;
        rigidbody.MovePosition(transform.position + move);
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
