import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

# Parallel environments
# env = gym.make('rj_gym_envs:bullets-v0')
# check_env(env, warn=True)

env = make_vec_env('rj_gym_envs:bullets-v0', n_envs=4)
model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)

# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading
# model = PPO.load("ppo_cartpole")

obs = env.reset()
env.render()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
