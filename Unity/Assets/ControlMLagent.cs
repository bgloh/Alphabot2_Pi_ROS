using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Random = UnityEngine.Random;

public class ControlMLagent : Agent
{
    public Boolean turnAgent = false;
    public Boolean randomAgentSize = false;
    private Vector3 startPosition;
    private Quaternion startRotation;
    new private Rigidbody rigidbody;
    private DuckieControl charController;
    // public float maxDistance = 1f;

    const int k_NoAction = 0; // do nothing!
    const int k_Up = 1;
    const int k_Down = 2;
    const int k_Left = 3;
    const int k_Right = 4;

    private LinkedList<GameObject> CheckPoint = new LinkedList<GameObject>();
    // private RaycastHit[] hits = new RaycastHit[5];
    // public LayerMask layerMask;
    // private int[] angle = {-90, -45, 0, 45, 90};
    // private float[] distance = {1.81f, 6f, 10f, 6f, 1.81f};

    public override void Initialize()
    {
        startPosition = transform.position;
        startRotation = transform.rotation;
        charController = GetComponent<DuckieControl>();
        rigidbody = GetComponent<Rigidbody>();
    }
    public override void OnEpisodeBegin()
    {
        // Reset agent position, rotation
        transform.position = startPosition;
        if (turnAgent)
        {
            transform.rotation = Quaternion.Euler(Vector3.up * Random.Range(-45f, 45f));   
        }
        else
        {
            transform.rotation = startRotation;
        }

        if (randomAgentSize)
        {
            transform.localScale = Vector3.one * Random.Range(0.5f, 1);
        }
        rigidbody.velocity = Vector3.zero;
        if (CheckPoint.Count != 0)
        {
            foreach (GameObject Check in CheckPoint)
            {
                Check.SetActive(true);
            }
            CheckPoint.Clear();
        }

    }
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        int vertical = Mathf.RoundToInt(Input.GetAxisRaw("Vertical"));
        int horizontal = Mathf.RoundToInt(Input.GetAxisRaw("Horizontal"));
        
        ActionSegment<int> actions = actionsOut.DiscreteActions;
        actions[0] = vertical >= 0 ? vertical : 2;
        actions[1] = horizontal >= 0 ? horizontal : 2;
        
        // var discreteActionsOut = actionsOut.DiscreteActions;
        // discreteActionsOut[0] = k_NoAction;
        // discreteActionsOut[1] = k_NoAction;
        // if (Input.GetKey(KeyCode.D))
        // {
        //     discreteActionsOut[0] = k_Right;
        // }
        // if (Input.GetKey(KeyCode.W))
        // {
        //     discreteActionsOut[0] = k_Up;
        // }
        // if (Input.GetKey(KeyCode.A))
        // {
        //     discreteActionsOut[0] = k_Left;
        // }
        // if (Input.GetKey(KeyCode.S))
        // {
        //     discreteActionsOut[0] = k_Down;
        // }
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        

        AddReward(-0.002f);
        var targetPos = transform.position;
        
        // var action = actions.DiscreteActions[0];
        // switch (action)
        // {
        //     case k_NoAction:
        //         charController.ForwardInput = 0;
        //         charController.TurnInput = 0;
        //         // do nothing
        //         break;
        //     case k_Right:
        //         charController.ForwardInput = 0;
        //         charController.TurnInput = 1;
        //         // targetPos = transform.position + new Vector3(1f, 0, 0f);
        //         break;
        //     case k_Left:
        //         // targetPos = transform.position + new Vector3(-1f, 0, 0f);
        //         charController.ForwardInput = 0;
        //         charController.TurnInput = -1;
        //         break;
        //     case k_Up:
        //         charController.ForwardInput = 1;
        //         charController.TurnInput = 0;
        //         // targetPos = transform.position + new Vector3(0f, 0, 1f);
        //         break;
        //     case k_Down:
        //         charController.ForwardInput = -1;
        //         charController.TurnInput = 0;
        //         // targetPos = transform.position + new Vector3(0f, 0, -1f);
        //         break;
        //     default:
        //         throw new ArgumentException("Invalid action value");
        // }
        
        float vertical = actions.DiscreteActions[0] <= 1 ? actions.DiscreteActions[0] : -1;
        float horizontal = actions.DiscreteActions[1] <= 1 ? actions.DiscreteActions[1] : -1;
        
        charController.ForwardInput = vertical;
        charController.TurnInput = horizontal;
        
        // for (int i = 0; i < hits.Length; i++)
        // {
        //     Vector3 direction = transform.forward;
        //     var quaternion = Quaternion.Euler(0, angle[i], 0);
        //     Vector3 newDirection = quaternion * direction;
        //
        //     if (Physics.Raycast(targetPos + (transform.forward * 1.1f) + (Vector3.down * 0.45f), newDirection, out hits[i], distance[i],
        //             layerMask))
        //     {
        //         Debug.DrawRay(targetPos + (transform.forward * 1.1f)+ (Vector3.down * 0.45f), newDirection * hits[i].distance, Color.blue);
        //     }
        //
        //     else { Debug.DrawRay(targetPos + (transform.forward * 1.1f)+ (Vector3.down * 0.45f), newDirection * distance[i], Color.red); }
        // }
        
        var hit = Physics.OverlapBox(
        targetPos, new Vector3(0.3f, 0.3f, 0.3f));
        // if (hit.Where(col => col.gameObject.CompareTag("Goal")).ToArray().Length == 1)
        // {
        //     SetReward(2f);
        //     EndEpisode();
        // }
        // else if (hit.Where(col => col.gameObject.CompareTag("Check")).ToArray().Length == 1)
        // {
        //     GameObject Check = hit.Where(col => col.gameObject.CompareTag("Check")).ToArray()[0].gameObject;
        //     float difangle = Math.Abs(Check.transform.eulerAngles.y - transform.eulerAngles.y);
        //     if (difangle > 20)
        //     {
        //         SetReward(1);
        //         // Debug.Log(1);
        //     }
        //     else
        //     {
        //         SetReward(2 - (difangle/20));
        //         // Debug.Log(2 - (difangle/20));
        //     }
        //     Check.SetActive(false);
        //     CheckPoint.AddLast(Check);
        // }
        if (hit.Where(col => col.gameObject.CompareTag("Line")).ToArray().Length == 1)
        {
            SetReward(-2f);
            EndEpisode();
        }
        // else if (hit.Where(col => col.gameObject.CompareTag("Out")).ToArray().Length == 1)
        // {
        //     SetReward(-2f);
        //     EndEpisode();
        // }
        // else if (hit.Where(col => col.gameObject.CompareTag("Car")).ToArray().Length == 1)
        // {
        //     SetReward(-3f);
        //     EndEpisode();
        // }
        
    }
    // public override void CollectObservations(VectorSensor sensor)
    // {
    //     for (int i = 0; i < 5; i++)
    //     {
    //         if (i == 2)
    //         {
    //             // Debug.Log(hits[i].distance * Math.Cos(Math.PI / 180 * 46));
    //             sensor.AddObservation((hits[i].distance * Mathf.Cos(Mathf.PI / 180 * 46)) / (distance[i] * Mathf.Cos(Mathf.PI / 180 * 46)));
    //         }
    //         else if(i == 1 || i == 3)
    //         {
    //             // Debug.Log(Math.Sqrt(Math.Pow(hits[i].distance / Math.Sqrt(2) * Math.Cos(Math.PI / 180 * 46), 2) + Math.Pow(hits[i].distance / Math.Sqrt(2), 2)));
    //             sensor.AddObservation(Mathf.Sqrt(Mathf.Pow(hits[i].distance / Mathf.Sqrt(2) * Mathf.Cos(Mathf.PI / 180 * 46), 2) + Mathf.Pow(hits[i].distance / Mathf.Sqrt(2), 2)) /
    //                                   Mathf.Sqrt(Mathf.Pow(distance[i] / Mathf.Sqrt(2) * Mathf.Cos(Mathf.PI / 180 * 46), 2) + Mathf.Pow(distance[i] / Mathf.Sqrt(2), 2)));
    //         }
    //         else
    //         {
    //             // Debug.Log(hits[i].distance);
    //             sensor.AddObservation(hits[i].distance);
    //         }
    //         // Debug.Log(hits[i].distance / distance[i]);
    //     }
    //
    // }
    // private void OnTriggerEnter(Collider other)
    // {
    //     // If the other object is a collectible, reward and end episode
    //     if (other.tag == "collectible")
    //     {
    //         AddReward(1f);
    //         EndEpisode();
    //     }
}
