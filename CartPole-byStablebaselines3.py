import gym
from stable_baselines3 import PPO

env = gym.make('CartPole-v1', render_mode="human")
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_cartpole")
env.close()