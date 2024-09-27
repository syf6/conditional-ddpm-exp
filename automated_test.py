import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import gymnasium as gym
import csv
import time
from gym.wrappers import RecordVideo

from tqdm import tqdm 

from itertools import product



class ConditionalDiffusionNet(nn.Module):
    def __init__(self, data_dim, cond_dim):
        super(ConditionalDiffusionNet, self).__init__()
        n_unit = 256

        self.l1 = nn.Linear(data_dim, n_unit)
        self.l2 = nn.Linear(n_unit, n_unit)

        self.l1_beta = nn.Linear(1, n_unit)
        self.l2_beta = nn.Linear(n_unit, n_unit)

        self.l1_cond = nn.Linear(cond_dim, n_unit)
        self.l2_cond = nn.Linear(n_unit, n_unit)

        self.l3 = nn.Linear(n_unit, n_unit)
        self.l4 = nn.Linear(n_unit, data_dim)

    def forward(self, x, c, t):
        xx = self.l1(x)
        xx = F.relu(xx)
        xx = self.l2(xx)
        xx = F.relu(xx)

        cc = self.l1_cond(c)
        cc = F.relu(cc)
        cc = self.l2_cond(cc)
        cc = F.relu(cc)

        bb = self.l1_beta(t)
        bb = F.relu(bb)
        bb = self.l2_beta(bb)
        bb = F.relu(bb)

        xx = self.l3(xx+bb+cc)
        xx = F.relu(xx)
        xx = self.l4(xx)

        return xx


class ConditionalDenoisingDiffusionProbabilisticModel():
    def __init__(self, X, cond, beta, device, batch_size=32):
        self.device = device

        self.X = X
        self.x_dim = self.X.shape[1]
        self.C = cond
        self.c_dim = self.C.shape[1]
        self.beta = beta
        self.n_beta = self.beta.shape[0]

        alpha = 1 - self.beta
        self.alpha = torch.tensor([[torch.prod(alpha[:i+1])] for i in range(self.n_beta)]).float()

        self.batch_size = batch_size

        self.model = ConditionalDiffusionNet(self.X.shape[1], self.C.shape[1]).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(self.X, self.C)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)



    def learning(self, n_epochs, lr):
        self.model.train()
        for e in range(n_epochs):
            for (x_batch, c_batch) in self.train_loader:
                loss_hist = []

                x_batch = x_batch
                c_batch = c_batch
                
                self.optimizer.zero_grad()

                t = torch.randint(low=0, high=self.n_beta, size=(x_batch.shape[0],))
                noise = torch.randn(x_batch.shape[0], self.x_dim)

                x_t = torch.sqrt(self.alpha[t]) * x_batch + torch.sqrt(1-self.alpha[t]) * noise

                noise_pred = self.model(x_t.to(self.device),
                                        c_batch.to(self.device),
                                        t[:,None].float().to(self.device))

                # Calculate loss
                loss = ((noise_pred - noise.to(self.device))**2).sum()
                loss_hist.append(loss.detach().cpu().numpy()/x_batch.shape[0])

                # Backpropagation
                loss.backward()

                # Update parameters
                self.optimizer.step()

            print('Epoch: {}, Loss: {}'.format(e, np.array(loss_hist).mean()))

        self.model.eval()



    def sampling(self, c, n=100):
        x_sample = torch.randn(n, self.x_dim)
        c_sample = c.repeat(n, 1)

        for t in range(self.n_beta)[::-1]:
            noise = torch.randn(n, self.x_dim)
            if t==0: noise= torch.zeros(n, self.x_dim)

            sigma = torch.sqrt(self.beta[t]*(1-self.alpha[t-1])/(1-self.alpha[t]))

            noise_pred = self.model(x_sample.to(self.device),
                                    c_sample.to(self.device),
                                    torch.tensor([[t]]).float().to(self.device)).detach().cpu()

            # import ipdb;ipdb.set_trace()
            x_sample = (x_sample - self.beta[t]*noise_pred/torch.sqrt(1-self.alpha[t])) / torch.sqrt(1-self.beta[t]) + sigma * noise


        return x_sample

def predict_and_execute(env, ddpm, device, n_steps, output_file = 'execution_log.csv'):
    # observation from gym evironment
    observation, _ = env.reset()

    # initialize a dataframe to log the data
    log_data =[]

    for step in range(n_steps):
        position = observation[0]
        velocity = observation[1]

        c_tensor = torch.tensor([[position, velocity]], dtype = torch.float32).to(device)

        with torch.no_grad():
            action_tensor = ddpm.sampling(c_tensor)
        
        action = action_tensor.cpu().numpy().flatten()

        # execute action in the environment
        observation, reward, done,_,_ = env.step(action)

        env.render()

        timestamp = time.time()

        # log data into csv file
        log_data.append({'timestamp':timestamp, 'action':action[0], 'position':position, 'velocity': velocity})

        if done:
            break

        time.sleep(0.1)
    
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(output_file , index = False) 

# 加载数据
file_path = 'episode_2.csv'
data = pd.read_csv(file_path)

# extraction: timestamp, action, position, velocity
x = data['action'].values[:, None]
c = data[['position', 'velocity']].values

# 定义超参数的可能值
learning_rates = [1e-3, 1e-4, 1e-5]
batch_sizes = [16, 32, 64]
n_epochs = [50, 100, 200]


beta = np.exp(np.linspace(np.log(0.001), np.log(0.9), 300))

# 使用 itertools.product 创建所有可能的超参数组合
hyperparameter_combinations = list(product(learning_rates, batch_sizes, n_epochs))

# 自动化测试超参数组合
for idx, (lr, batch_size, n_epochs) in enumerate(tqdm(hyperparameter_combinations, total=len(hyperparameter_combinations))):
    # 创建一个包含所有超参数的字符串，用于文件命名
    lr_str = f"lr_{lr:.0e}"
    batch_size_str = f"batch_{batch_size}"
    n_epochs_str = f"epochs_{n_epochs}"
    
    param_str = f"{lr_str}_{batch_size_str}_{n_epochs_str}"
    
    # 使用超参数字符串创建输出文件名
    output_file = f'execution_log_{param_str}.csv'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 创建 DDPM 模型实例
    ddpm = ConditionalDenoisingDiffusionProbabilisticModel(
        torch.tensor(x).float(),
        torch.tensor(c).float(),
        torch.tensor(beta).float(), device, batch_size=batch_size)
    
    # 训练模型
    ddpm.learning(n_epochs=int(n_epochs), lr=lr)
    
    # 预测并执行
    env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array")
    predict_and_execute(env, ddpm, device, n_steps=1000, output_file=output_file)
    env.close()