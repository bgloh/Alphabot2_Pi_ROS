import numpy as np
from mlagents_envs.environment import UnityEnvironment as UE
import onnxruntime

env = UE(file_name='../DuckieBotProject.exe', seed=1, side_channels=[]) # 현재 생성된 환경 불러오기
model = "Test.onnx"
env.reset()
behavior_name = list(env.behavior_specs)[0] # 행동 이름 받아오기
#
decision_steps, terminal_steps = env.get_steps(behavior_name)
sess = onnxruntime.InferenceSession(model)
order = (0, 3, 1, 2)
print(decision_steps.obs[0].shape)
print(decision_steps.action_mask)
output = sess.run(["discrete_actions"], {"obs_0": np.array(decision_steps.obs[0]).transpose(order), "action_masks": np.array([[1., 1., 1., 1., 1.]]).astype(np.float32)})
print(output)

sess.get_modelmeta()
first_input_name = sess.get_inputs()[0].name
first_output_name = sess.get_outputs()[0].name
print(sess.get_inputs()[0])
print(sess.get_inputs()[1])
# print()
env.close()


# print(behavior_name) # 출력
# spec = env.behavior_specs[behavior_name] # 얻은 이름을 바탕으로 관측치
#
# print("Number of observations : ", len(spec.observation_specs)) # 관측치의 종류
#
# vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
# print("Is there a visual observation ?", vis_obs) # 영상 정보인지 확인
#
# for episode in range(10):
#   env.reset()
#   decision_steps, terminal_steps = env.get_steps(behavior_name)
#   tracked_agent = -1 # -1 indicates not yet tracking
#   done = False # For the tracked_agent
#   episode_rewards = 0 # For the tracked_agent
#   while not done:
#     # Track the first agent we see if not tracking
#     # Note : len(decision_steps) = [number of agents that requested a decision]
#     if tracked_agent == -1 and len(decision_steps) >= 1:
#       tracked_agent = decision_steps.agent_id[0]
#     # vector_observation:0

#     # Generate an action for all agents
#     action = spec.action_spec.random_action(len(decision_steps))
#
#     # Set the actions
#     env.set_actions(behavior_name, action)
#
#     # Move the simulation forward
#     env.step()
#
#     # Get the new simulation results
#     decision_steps, terminal_steps = env.get_steps(behavior_name)
#     if tracked_agent in decision_steps: # The agent requested a decision
#       episode_rewards += decision_steps[tracked_agent].reward
#     if tracked_agent in terminal_steps: # The agent terminated its episode
#       episode_rewards += terminal_steps[tracked_agent].reward
#       done = True
#   print(f"Total rewards for episode {episode} is {episode_rewards}")