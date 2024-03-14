import gym
env = gym.make('MountainCar-v0')
for episode in range(10):
    env.reset()
    print("Episode finished after {} timesteps".format(episode))
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())

env.close()