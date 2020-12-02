from gym.envs.registration import register

register(
    id='bullets-v0',
    entry_point='rj_gym_envs.envs:BulletsEnv',
)
register(
    id='bullets-simple-v0',
    entry_point='rj_gym_envs.envs:BulletsSimpleEnv',
)
# register(
#     id='cartpole-v0',
#     entry_point='rj_gym_envs.envs:CartpoleEnv',
# )
