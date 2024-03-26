from random import random

import gym
import numpy as np

# Initialize env
env = gym.make(id='Pendulum-v1')
STATE_DIM = env.observation_space.shape[0]  #读取状态的维度
ACTION_DIM = env.action_space.shape[0]  #读取动作的维度，便于神经网络输入输出的搭建

agent = DDPGAgent(STATE_DIM, ACTION_DIM)    #TODO

# Hyperparameters(超参数)
NUM_EPISODE = 100  #一共100局
NUM_STEP = 200     #因为倒立摆每一局就200步，不同环境有不同的步数
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000   #一共20000步，后一万步只有百分之二的探索率

for episode_i in range(NUM_EPISODE):
    state, others = env.reset()

    for step_i in range(NUM_STEP):
        epsilon = np.interp(x=episode_i*NUM_STEP+step_i, xp=[0, EPSILON_DECAY], fp=[EPSILON_START, EPSILON_END])    #规定探索方式
        random_sample = random.random() #产生[0.0,1.0)之间的随机浮点数
        if random_sample <= epsilon:
            action = np.random.uniform(low=-2, high=2, size=ACTION_DIM) #由于是一个一维的动作空间，按照均匀分布随机取样
        else:
            action = agent.get_action(state)    #确定性的方式

        env.step(action)
