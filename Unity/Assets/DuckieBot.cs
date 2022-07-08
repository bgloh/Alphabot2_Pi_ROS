using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

using Vector3 = UnityEngine.Vector3;

public class DuckieBot : Agent
{
    Rigidbody rBody;
    private DuckieControl duckieControl;
   
    // forceMultiplier는 메소드가 정의되기 전에 정의 되서 Unity의 인스팩터 윈도우에서 변경 가능하다.
    public float forceMultiplier = 10;
   
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
        duckieControl = GetComponent<DuckieControl>();
    }

    public Transform Target;
   
    // 에피소드가 시작되었을 때(이전 에피소드가 모종의 이유로 종료되었을 경우)
    public override void OnEpisodeBegin()
    {
        // 에이전트가 떨어졌을 경우 작동
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }
        // 타겟의 위치를 랜덤하게 이동
        Target.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        // Target의 위치 좌표 x,y,z 3개
        sensor.AddObservation(this.transform.localPosition);
        // Agent의 위치 좌표 x,y,z 3개
       
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
        // Agent의 속도 y가 없는이유는 이 에이전트는 y방향으로는 이동하지 않기 때문
    }

    public override void OnActionReceived(ActionBuffers action)
    {
        float vertical = action.DiscreteActions[0] <= 1 ? action.DiscreteActions[0] : -1;
        float horizontal = action.DiscreteActions[1] <= 1 ? action.DiscreteActions[1] : -1;

        duckieControl.ForwardInput = vertical;
        duckieControl.TurnInput = horizontal;
        // // 2차원 액션
        // Vector3 controlSignal = Vector3.zero;// 3차원 0벡터를 생성
        // controlSignal.x = action.ContinuousActions[0];// x 방향으로의 백터 값 추가
        // controlSignal.z = action.ContinuousActions[1];// z 방향으로의 백터 값 추가
        // rBody.AddForce(controlSignal * forceMultiplier);// x, z 방향으로의 백터값의 힘을 곱하여 리지드 바디에 추가한다.

        // float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);
        // // 타겟과 에이전트의 거리 계산
        //
        // // 목표에 도달했을 경우
        // if (distanceToTarget < 1.42f)
        // {
        //     SetReward(1.0f);// 보상 설정 후
        //     EndEpisode();// 에피소드 종료 -> 초기화
        // }
       
        // // 플렛폼 밖으로 떨어졌을 경우
        // else if (this.transform.localPosition.y < 0)
        // {
        //     EndEpisode();// 에피소드 종료 -> 초기화
        // }
    }
   
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        int vertical = Mathf.RoundToInt(Input.GetAxisRaw("Vertical"));
        int horzontal = Mathf.RoundToInt(Input.GetAxisRaw("Horizontal"));
        ActionSegment<int> actions = actionsOut.DiscreteActions;
        actions[0] = vertical >= 0 ? vertical : 2;
        actions[1] = horzontal >= 0 ? horzontal : 2;
    }
}