print("starting gym import", flush=True)
import gym
print("finishing gym import, starting gym.make", flush=True)
env = gym.make('HalfCheetah-v2')
print("finishing gym.make, starting env.reset", flush=True)
env.reset()
print("finishing env.reset, script complete", flush=True)