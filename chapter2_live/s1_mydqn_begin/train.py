import gym
import torch
from chapter2_live.s1_mydqn_begin import agents,modules
class TrainManger():
    def __init__(self ,env, episodes=1000, lr=0.001, gamma=0.9, e_greed=0.1):
        self.env = env
        self.episodes = episodes
        n_obs=env.observation_space.shape[0]
        n_act=env.action_space.n
        q_func = modules.MLP(n_obs, n_act)
        optimizer = torch.optim.Adam(q_func.parameters(), lr=lr)
        self.agent = agents.DQNAgent(
            q_func=q_func,
            optimizer=optimizer,
            n_act=n_act,
            gamma=gamma,
            e_greed=e_greed
        )

    def train_episode(self):    #需要反向传播
        total_reward = 0    #看看运行的效果
        obs = self.env.reset()
        obs = torch.FloatTensor(obs)
        while True:
            action = self.agent.act(obs)
            next_obs,reward,done,_ = self.env.step(action) #与环境交互并生成一系列值
            next_obs = torch.FloatTensor(next_obs)
            self.agent.learn(obs,action,reward,next_obs,done)    #训练起来

            obs = next_obs      #形成循环
            total_reward += reward
            if done:
                break
        return total_reward

    def test_episode(self): #不需要反向传播，is_render参数用来判断是否需要渲染
        total_reward = 0  # 看看运行的效果
        obs = self.env.reset()
        obs = torch.FloatTensor(obs)
        while True:

            action = self.agent.act(obs)
            next_obs, reward, done, _ = self.env.step(action)  # 与环境交互并生成一系列值
            next_obs = torch.FloatTensor(next_obs)
            obs = next_obs  # 形成循环
            total_reward += reward
            self.env.render()
            if done:break
        return total_reward

    def train(self):    #训练
        for e in range(self.episodes):
            ep_reward = self.train_episode()  #每一次的reward
            print('Episode %s: reward= %.1f' %(e,ep_reward))
            if e%100 == 0:
                test_reward = self.test_episode()
                print('test_reward= %.1f' % (test_reward))

if __name__ == '__main__':
    env1 = gym.make('CartPole-v0')
    tm = TrainManger(env1)
    tm.train()  #把环境传进去