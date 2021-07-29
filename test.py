print("started bcq imports", flush=True)
import argparse
print("started mujoco_py import", flush=True)
import mujoco_py
print("finished mujoco_py import", flush=True)
print("started gym import", flush=True)
import gym
print("finished gym import", flush=True)
import numpy as np
import os

os.system("pip list | grep mujoco-py ")

print("started torch import", flush=True)
import torch
print("finished torch import", flush=True)
#import wandb
print("started d4rl import", flush=True)
import d4rl
print("finished d4rl import", flush=True)
import uuid
import json
print("finished non-model bcq imports", flush=True)
