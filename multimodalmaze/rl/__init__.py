
from gym.envs.registration import registry, register, make, spec

register(
    id='MultimodalMaze-v0',
    entry_point='multimodalmaze.rl.mmaze_env:MultimodalMazeEnv',
)