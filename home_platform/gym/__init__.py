from gym.envs.registration import register

register(
    id='Home-v0',
    entry_point='home_platform.gym.envs:HomeEnv',
    max_episode_steps=1000,
    reward_threshold=0.0,
)