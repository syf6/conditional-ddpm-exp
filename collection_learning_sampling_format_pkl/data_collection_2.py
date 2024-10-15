import sys, os
import numpy as np
import gym
import pygame

import pickle as pkl

if __name__=='__main__':

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

        # initially this is 500
        for e in range(400):
            states = []
            actions = []
        
            observation, info = env.reset()
            
            for i in range(1000):
                # env.render()
            
                # action = env.action_space.sample()
                axis_values = [joystick.get_axis(i) for i in range(num_axes)]
                # action = [axis_values[0]] 
                action = np.array([axis_values[0]])
                observation, reward, terminated, truncated, info = env.step(action)

                states.append(observation)
                actions.append(action)
            
                # print(i, observation, action, reward, terminated)
            
                if terminated or truncated:
                    break
                    # observation, info = env.reset()
                
            print(e)

            data_state.append(np.array(states))
            data_action.append(np.array(actions))
        
        env.close()

        os.makedirs('./data', exist_ok=True)

        with open('./data/train_state.pkl', 'wb') as f:
            pkl.dump(data_state, f)
        with open('./data/train_action.pkl', 'wb') as f:
            pkl.dump(data_action, f)