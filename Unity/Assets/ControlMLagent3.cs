using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using Random = UnityEngine.Random;

public class ControlMLagent3 : Agent
{
    public Boolean turnAgent = false;
    public Boolean randomAgentSize = false;
    private Vector3 startPosition;
    private Quaternion startRotation;
    new private Rigidbody rigidbody;
    private DuckieControl2 charController;
    // public float maxDistance = 1f;
    
    private LinkedList<GameObject> CheckPoint = new LinkedList<GameObject>();
    private LinkedList<GameObject> OverPoint = new LinkedList<GameObject>();

    public override void Initialize()
    {
        startPosition = transform.position;
        startRotation = transform.rotation;
        charController = GetComponent<DuckieControl2>();
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
        if (OverPoint.Count != 0)
        {
            foreach (GameObject Over in OverPoint)
            {
                Over.SetActive(true);
            }
            OverPoint.Clear();
        }

    }
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        int vertical = Mathf.RoundToInt(Input.GetAxisRaw("Vertical"));
        int horizontal = Mathf.RoundToInt(Input.GetAxisRaw("Horizontal"));
        
        ActionSegment<int> actions = actionsOut.DiscreteActions;
        actions[0] = vertical >= 0 ? vertical : 2;
        actions[1] = horizontal >= 0 ? horizontal : 2;
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        

        AddReward(-0.002f);
        var targetPos = transform.position;
        
        
        float vertical = actions.DiscreteActions[0] <= 1 ? actions.DiscreteActions[0] : -1;
        float horizontal = actions.DiscreteActions[1] <= 1 ? actions.DiscreteActions[1] : -1;
        
        charController.ForwardInput = vertical;
        charController.TurnInput = horizontal;

        var hit = Physics.OverlapBox(
        targetPos, new Vector3(0.3f, 0.3f, 0.3f));
        // if (hit.Where(col => col.gameObject.CompareTag("Goal")).ToArray().Length == 1)
        // {
        //     SetReward(2f);
        //     EndEpisode();
        // }
        if (hit.Where(col => col.gameObject.CompareTag("Check")).ToArray().Length == 1)
        {
            GameObject Check = hit.Where(col => col.gameObject.CompareTag("Check")).ToArray()[0].gameObject;
            float difangle = Math.Abs(Check.transform.eulerAngles.y - transform.eulerAngles.y);
            if (difangle > 20)
            {
                SetReward(1);
                // Debug.Log(1);
            }
            else
            {
                SetReward(2 - (difangle/20));
                // Debug.Log(2 - (difangle/20));
            }
            Check.SetActive(false);
            CheckPoint.AddLast(Check);
        }
        else if (hit.Where(col => col.gameObject.CompareTag("Line")).ToArray().Length == 1)
        {
            SetReward(-2f);
            EndEpisode();
        }
        // else if (hit.Where(col => col.gameObject.CompareTag("Out")).ToArray().Length == 1)
        // {
        //     SetReward(-2f);
        //     EndEpisode();
        // }
        else if (hit.Where(col => col.gameObject.CompareTag("Car")).ToArray().Length == 1)
        {
            SetReward(-3f);
            EndEpisode();
        }
        else if (hit.Where(col => col.gameObject.CompareTag("Over")).ToArray().Length == 1)
        {
            GameObject Over = hit.Where(col => col.gameObject.CompareTag("Over")).ToArray()[0].gameObject;
            Over.SetActive(false);
            OverPoint.AddLast(Over);
            // Debug.Log("reward");
            SetReward(3f);
        }
        
    }
}
