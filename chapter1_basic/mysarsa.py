import time
import numpy as np
import gym  #引入的强化学习环境
import gridworld


class SarsaAgent():

    def __init__(self,n_states,n_act,e_greed=0.1,lr=0.1,gamma=0.9):  #初始化Q表格要用到n_states和n_act；e_greed代表探索概率
        self.e_greed=e_greed
        self.Q = np.zeros([n_states,n_act]) #初始化Q表格为全为0
        self.n_act = n_act
        self.n_states = n_states
        self.lr = lr
        self.gamma = gamma

    def predict(self,state):  #利用之前Q表格内的信息，传进来当的状态
        Q_list = self.Q[state,:]    #做切片，取出state对应的那一排
        #action = np.argmax(Q_list)   #取出最大值，但仅会采取第一个最大值对应的下标，比如[1,1,1,1]，返回0
        action = np.random.choice(np.flatnonzero(Q_list==Q_list.max())) #这样即可随机选取所有的最大值对应的下标
        return action
    def act(self,state):    #state代表当前环境的状态
        if np.random.uniform(0,1)<self.e_greed: #完全随机地探索
            action = np.random.choice(self.n_act)
        else:   #利用Q表格中已存在的信息
            action = self.predict(state)
        return action   #返回要取用的action


    def learn(self,state,action,reward,next_state,next_action,done):    #反向传播,更准确地说就是td算法
        cur_Q = self.Q[state,action]
        if done:
            target_Q = reward
        else:
            target_Q = reward + self.gamma*self.Q[next_state,next_action]   #SARSA算法公式中目标值部分
        self.Q[state,action] += self.lr * (target_Q - cur_Q)

def train_episode(env,agent,is_render):    #需要反向传播
    total_reward = 0    #看看运行的效果
    state = env.reset()
    action = agent.act(state)

    while True:
        next_state,reward,done,_ = env.step(action) #与环境交互并生成一系列值
        next_action = agent.act(next_state) #进行探索机制

        agent.learn(state,action,reward,next_state,next_action,done)    #训练起来

        action = next_action
        state = next_state      #形成循环
        total_reward += reward
        if is_render:env.render()
        if done:
            break
    return total_reward

def test_episode(env,agent): #不需要反向传播，is_render参数用来判断是否需要渲染
    total_reward = 0  # 看看运行的效果
    state = env.reset()

    while True:

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)  # 与环境交互并生成一系列值

        state = next_state  # 形成循环
        total_reward += reward
        env.render()
        time.sleep(0.5)
        if done:break
    return total_reward

def train(env,episode=500,e_greed=0.1,lr=0.1,gamma=0.9):    #训练
    agent = SarsaAgent(
        n_states=env.observation_space.n,
        n_act=env.action_space.n,
        lr=lr,
        gamma=gamma,
        e_greed=e_greed
    )
    is_render = False
    for e in range(episode):
        ep_reward = train_episode(env,agent,is_render)  #每一次的reward
        print('Episode %s: reward= %.1f' %(e,ep_reward))

        if e % 50 == 0:
            is_render = True
        else:
            is_render = False
    test_reward = test_episode(env,agent)
    print('test_reward= %.1f' % (test_reward))

if __name__ == '__main__':
    env = gym.make('CliffWalking-v0')
    env = gridworld.CliffWalkingWapper(env)
    train(env)  #把环境传进去