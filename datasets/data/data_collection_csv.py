import sys, os
import numpy as np
import gym
import pygame
import csv

if __name__ == '__main__':
    os.makedirs('./data', exist_ok=True)

    # env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array")
    env = gym.make('MountainCarContinuous-v0', render_mode="human")

    pygame.init()
    pygame.joystick.init()

    data_state = []
    data_action = []

    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"initialized joystick: {joystick.get_name()}")

        num_axes = joystick.get_numaxes()
        print(f"number of axes: {num_axes}")

        for e in range(2):
            states = []
            actions = []
        
            observation, info = env.reset()
            
            for i in range(1000):
                # env.render()
            
                # 读取摇杆轴值并作为动作
                axis_values = [joystick.get_axis(i) for i in range(num_axes)]
                action = np.array([axis_values[0]])  # 使用第一个轴的值作为动作
                observation, reward, terminated, truncated, info = env.step(action)

                # 记录状态和动作
                states.append(observation)
                actions.append(action)
            
                if terminated or truncated:
                    break
                
            print(f"Episode {e+1} completed")

            data_state.append(np.array(states))
            data_action.append(np.array(actions))
        
        env.close()

        # 所有 episode 完成后，保存到一个 CSV 文件
        with open('./data/manual_policy_2_RLR.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(["episode", "step", "position", "velocity", "action"])

            # 遍历所有 episode 和它们的 step，逐行写入数据
            for episode_idx, (episode_states, episode_actions) in enumerate(zip(data_state, data_action)):
                for step_idx, (state, action) in enumerate(zip(episode_states, episode_actions)):
                    position, velocity = state  # 拆分状态为位置和速度
                    writer.writerow([episode_idx + 1, step_idx + 1, position, velocity, action[0]])

        print("All data saved to combined_data.csv.")
