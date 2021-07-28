print("starting mujoco-py import", flush=True)
import mujoco_py
print("finishing mujoco-py import, starting gym import", flush=True)
import gym
print("finishing gym import, starting gym.make", flush=True)
env = gym.make('HalfCheetah-v2')
print("finishing gym.make, starting env.reset", flush=True)
env.reset()
print("finishing env.reset, starting d4rl import", flush=True)
import d4rl
print("finishing d4rl import, script complete", flush=True)
