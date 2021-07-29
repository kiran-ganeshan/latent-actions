import mujoco_py
import d4rl
import gym
datasets = ["expert-v0", "medium-v0", "random-v0"]
envs = ["halfcheetah"]
for env in envs:
    for dataset in datasets:
        e = gym.make(env + "-" + dataset)
        d = d4rl.qlearning_dataset(e)