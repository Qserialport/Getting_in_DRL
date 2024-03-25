import random
import gym
import numpy as np
import torch
import torch.nn as nn
from agent import Agent

env = gym.make('CartPole-v1')
s = env.reset() #初始化并输出一系列观测值

TARGET_UPDATE_FREQUENCY = 10    #每十局更新一次Q网络参数

EPSILON_DECAY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.02         #贪心算法参数会逐渐减小

n_episode = 5000   #一共玩5000局
n_time_step = 1000  #每一局有1000步

n_state = len(s)
n_action = env.action_space.n

agent = Agent(n_input=n_state, n_output=n_action)

REWARD_BUFFER = np.empty(shape=n_episode)
for episode_i in range(n_episode):
    episode_reward = 0      #每个阶段的总体奖励
    for step_i in range(n_time_step):
        epsilon = np.interp(episode_i * n_time_step + step_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END]) #一维线性插值
        random_sample = random.random()

        if random_sample <= epsilon:            #根据epsilon选择将要执行的动作
            a = env.action_space.sample()
        else:
            a = agent.online_net.act(s)  # TODO

        s_, r, done, info = env.step(a)     #执行动作a，并观测下一状态、奖励、是否结束
        agent.memo.add_memo(s, a, r, done, s_)  # TODO  ,加入经验池
        s = s_
        episode_reward += r

        if done:
            s = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            break

        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()   #TODO   #每个batch由四元组组成（s,r,a,s_）

        # Compute target using td_AL
        target_q_values = agent.target_net(batch_s_)    #神经网络输出为一个张量，注意使用的是下一时刻的状态
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = batch_r + agent.GAMMA * (1-batch_done) * max_target_q_values   #TODO

        # Compute q_values
        q_values = agent.online_net(batch_s)   #TODO       #注意使用的是现在时刻的状态，但是注意，这个batch_s不止一个状态，每一个状态均要输入nn中，并输出每一个动作对应的Q值
        a_q_values = torch.gather(input=q_values, dim=1, index=batch_a) #找出每个状态，对应的所有Q值里面最大的，并输出

        # Compute loss
        loss = nn.functional.smooth_l1_loss(targets, a_q_values)

        # Gradient descent
        agent.optimizer.zero_grad()   #TODO 要输入状态
        loss.backward() #反向传播
        agent.optimizer.step()  #至此完成神经网络的一次梯度下降   #TODO

    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())   #TODO

        # Show the training process
        print("Episode: {}".format(episode_i))
        print("Average Reward: {}".format(np.mean(REWARD_BUFFER[:episode_i])))  #直到现在的平均奖励


