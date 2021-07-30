import os
import argparse
import numpy as np
if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument('program', type=str, default='bcq', help='Name of program')
    parser.add_argument('env', type=str, default='halfcheetah-expert-v0', help='Name of environment')
    parser.add_argument('seed', type=int, default=0, help='Starting seed to pull experiment data')
    parser.add_argument('N', type=int, default=1, help='Number of experiments to pull')
    args = parser.parse_args()
    program = args.program
    env = args.env
    seed = args.seed
    lst = list()
    old_r = np.load('./outputs/{program}_{env}_all.npy')
    for n in range(N):
        os.system(f"gsuitl cp gs://kiran-research/latent-actions/outputs/{program}/seed{seed + n}/{program}_{env}_{seed}.npy ./outputs/{program}_{env}/{seed}.npy")
        lst.append(np.load('./outputs/{program}_{env}/{seed}'))
    r = np.array(lst)
    r = np.concatenate(old_r, r, axis=0)
    np.save(r, './outputs/{program}_{env}_all.npy')
