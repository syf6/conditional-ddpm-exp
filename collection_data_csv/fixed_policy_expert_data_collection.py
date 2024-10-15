import sys, os
import numpy as np
import gym
import csv

def action_decision(obs):
    pos, vel = obs
    # action = np.array([0.5 * vel + 0.005 * (0.5 - pos)]) * 10
    # return np.clip(action, -1, 1)

    if vel > 0:
        action = np.array([1.0])
    else:
        action = np.array([-1.0])

    return action
    

if __name__ == '__main__':

    os.makedirs('./data', exist_ok=True)

    # env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array")
    env = gym.make('MountainCarContinuous-v0') # render_mode="human" can be added if required

    data_state = []
    data_action = []

    for e in range(2):
        states = []
        actions = []
    
        observation, info = env.reset()
        
        for i in range(1000):
            # env.render()
        
            action = action_decision(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            states.append(observation)
            actions.append(action)
        
            if terminated or truncated:
                break

        data_state.append(np.array(states))
        data_action.append(np.array(actions))
    
    env.close()

    # 保存所有数据到 CSV 文件
    os.makedirs('./data', exist_ok=True)
    with open('./data/fixed_policy_data_2.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(["episode", "step", "position", "velocity", "action"])

        # 遍历所有 episode 和它们的 step，逐行写入数据
        for episode_idx, (episode_states, episode_actions) in enumerate(zip(data_state, data_action)):
            for step_idx, (state, action) in enumerate(zip(episode_states, episode_actions)):
                position, velocity = state  # 拆分状态为位置和速度
                writer.writerow([episode_idx + 1, step_idx + 1, position, velocity, action[0]])

    print("All data saved to csv.")
