import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.env_checker import check_env

training_mode = True

# Single env
env = gym.make('rj_gym_envs:bullets-v0')

# Parallel envs
# env = make_vec_env('rj_gym_envs:bullets-v0', n_envs=4)

model = PPO(MlpPolicy, env, verbose=1)
if training_mode:
    model.learn(total_timesteps=25000)
    model.save("net/ppo_bullets")
else:
    model.load("net/ppo_bullets")

obs = env.reset()
steps = 0
while steps < 10000:
    steps += 1
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    print(f'Step: {steps}  Player HP: {env.player_ship.hp}  Boss HP: {env.boss_ship.hp}')

print('Done')
