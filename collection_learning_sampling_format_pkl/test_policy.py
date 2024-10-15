import sys, os
import numpy as np
import gym
import pygame

import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from learning_ddpm import ConditionalDiffusionNet, ConditionalDenoisingDiffusionProbabilisticModel


if __name__=='__main__':

    with open('./data/learned_policy.pkl', 'rb') as f:
        policy_ddpm = pkl.load(f)


    # env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array")
    env = gym.make('MountainCarContinuous-v0', render_mode="human")

    data_state = []
    data_action = []

    for e in range(5):
        states = []
        actions = []
    
        observation, info = env.reset()
        
        for i in range(1000):
            # env.render()
        
            # action = env.action_space.sample()
            action = policy_ddpm.sampling(torch.tensor(observation).float(), n=1)
            action = action[0].numpy()

            observation, reward, terminated, truncated, info = env.step(action)

            states.append(observation)
            actions.append(action)
        
            # print(i, observation, action, reward, terminated)
        
            if terminated or truncated:
                break
                # observation, info = env.reset()

        data_state.append(np.array(states))
        data_action.append(np.array(actions))
    
    env.close()

