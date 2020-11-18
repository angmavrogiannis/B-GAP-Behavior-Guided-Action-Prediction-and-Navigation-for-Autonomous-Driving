import gym
import highway_env

env = gym.make("highway-v0")
env.reset()
done = False
count = 0
while not done:
    action = 1
    output = env.step(action)
    # print(output[1][1], output[1][1][1])
    # input('a')
    env.render()
    if (count == 1000):
        done = True
    count += 1