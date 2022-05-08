# MLAgent

[깃헙](https://github.com/k-chan-l/Alphabot2_Pi_ROS/tree/AI-robot-2022/test)


## MLAgent 폴더 내부에 관련 파일 존재

중요 파일들만 따로 설명

DuckieBotProject.exe

유니티로 빌드된 프로젝트 파일 더블클릭시 실행 가능하며, wasd로 조작가능


![alt_text](https://user-images.githubusercontent.com/71301248/167301641-652abaa8-8ef9-42d3-8f27-8ce387de0483.png)



## python 폴더

MLAgent 환경을 사용하는 파이썬 코드들이 들어있다.


### ml-agent.py

이 파일은 ml-agent를 통해서 빌드된 환경을 다룰 수 있는 파이썬 파일이다. 기본적인 mlagent python toolkit을 보여주며 랜덤한 액션을 주어서 테스트 할 수 있는 파일이다.


### reinforcement.py

이 파일을 통해 파이썬 코드를 통한 학습을 테스트 해보았다. 네트워크로는 qnet을 사용한다.

인풋정보로는 이미지와 action mask를 받는다.


![alt_text](https://user-images.githubusercontent.com/71301248/167301651-7d65e47b-849c-4882-b24c-52205b0edc29.png)


이 부분을 수정하여 학습 성능을 향상 시킬 수 있다. 현재는 빠른 테스트를 위해서 training step과 exp를 줄여놓았다. 실행 완료시 최종적으로 Test2.onnx파일을 생성한다.


### Test.onnx

사전 학습된 파일이다. 위의 reinforcement.py를 통해 학습한 모델이다.


### onnxtest.py

Test.onnx를 사용하는 파일이다. 핵심 코드는


```
sess = onnxruntime.InferenceSession(model)
order = (0, 3, 1, 2)
output = sess.run(["discrete_actions"], {"obs_0": np.array(decision_steps.obs[0]).transpose(order), "action_masks": np.array([[1., 1., 1., 1., 1.]]).astype(np.float32)})
```


이 3줄의 코드이다.

첫번째 sess는 onnx파일을 불러서 실행하는 코드이다.

두번째 order는 decision_steps.obs[0]을 순서를 바꾸는 코드이다. 여기서는 영상이 (1, 480, 640, 3)으로 주어지는데 이 정보를 (1, 640, 3, 480)으로 바꾸는 코드이다.

여기서 1은 에이전트, 480은 세로, 640은 가로, 3은 채널 수를 의미한다.

세번째 output은 정보들을 바탕으로 코드를 실행하는 것이다. action_mask 부분의 타입을 np.float32로 지정 해줘야 신경망 내부에서 타입이 변환되지 않는다.

해당 코드를 실행하면 output 결과 값을 출력한다.


### ml–agentwithonnx.py

Test.onnx를 이용하여 에이전트를 조작하는 코드이다.


```
action = spec.action_spec.empty_action(len(decision_steps))
action.add_discrete(output[0])
env.set_actions(behavior_name, action)
```


여기서 중요한 코드는 이 3코드이다.

먼저 첫번째 줄의 코드는 spec이라는 환경에서 얻은 객체에서 액션을 나타내는 action_spec 그중에서 비어있는 empty_action을 액션의 총 길이 만큼 생성한다.

두번째 줄은 액션에 discrete action을 더하는 코드인데 discrete action인 이유는 해당 에이전트가 물리엔진이 아닌 좌표값으로 이동해서 discrete action을 사용한다. output[0]는 onnx파일로 추론한 정보를 사용하는 것인데 output의 경우에는 [0]에 추론된 정보를, [1]에 해당 타입을 가지고 있는 list 형식의 객체이다.

add_discrete의 경우는 덮어 씌워지는 함수라서 뒤에 같은 함수가 호출되면 앞의 값은 사라진다. 이 코드 대신에


```
action.add_discrete(np.array([[1]]))
```


이걸 사용 할 수도 있는데 이걸 통해서 액션을 조작 할 수 있다.

세번째 줄은 생성된 액션을 환경에 집어넣어서 수행하는 코드이다.

이 코드를 통해 output에서 얻어낸 값들이 어떤것을 의미하는 지 알 수 있다.

0 : 정지, 1 : 전진, 2 : 후진, 3 : 좌회전, 4 : 우회전 이다.


### onnxtestImage.py

이미지(TestImage.jpg)를 통해 onnx를 테스트 하는 코드이다.

resize 부분과 transpose부분이 있는데 위의 코드의 transpose부분이 정상적으로 작동되지 않아서 그부분을 제외하고 실행했더니 기대한 결과가 나왔다.

이유는 이미지가 다르기 때문으로 추측중이다.
