import time
import numpy as np
import torch


class DQNAgent():

    def __init__(self, q_func,  optimizer,n_act, e_greed=0.1, gamma=0.9):  #初始化Q表格要用到n_states和n_act；e_greed代表探索概率
        self.q_func = q_func
        self.optimizer = optimizer    #优化器
        self.criterion = torch.nn.MSELoss() #初始化平方差损失函数
        self.e_greed=e_greed

        self.n_act = n_act
        self.gamma = gamma

    def predict(self,obs):  #利用之前Q表格内的信息，传进来当的状态
        Q_list = self.q_func(obs)    #做切片，取出state对应的那一排
        #action = np.argmax(Q_list)   #取出最大值，但仅会采取第一个最大值对应的下标，比如[1,1,1,1]，返回0
        action = int(torch.argmax(Q_list).detach().numpy()) #torch出来的是张量形式，要将其detach掉转换为numpy形式，再转换成小数
        return action
    def act(self,obs):    #state代表当前环境的状态
        if np.random.uniform(0,1)<self.e_greed: #完全随机地探索
            action = np.random.choice(self.n_act)
        else:   #利用Q表格中已存在的信息
            action = self.predict(obs)
        return action   #返回要取用的action


    def learn(self,obs,action,reward,next_obs,done):    #反向传播
        cur_Q = self.q_func(obs)[action]
        target_Q = reward + ( 1 - float(done) ) * self.gamma * self.q_func(next_obs).max()   #SARSA算法公式中目标值部分

        #更新参数
        self.optimizer.zero_grad()  #梯度归零
        loss = self.criterion(cur_Q, target_Q)
        loss.backward() #反向传播
        self.optimizer.step()