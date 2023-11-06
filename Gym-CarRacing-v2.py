import gym
env = gym.make("CarRacing-v2", render_mode="human")
observation, info = env.reset()

while True:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()