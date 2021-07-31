from google.cloud import storage
import numpy as np
import os, sys
from matplotlib import pyplot as plt
            
'''
Get and compile GCP return data.
Usage: python download_returns.py {program} {env} {start_seed} {N}
Example: To download returns from bcq on halfcheetah-expert-v0 for seeds 1950-1959,
python download_returns.py bcq halfcheetah-expert-v0 1950 10
'''

def get_item_data(item, N, seed, program, env):
    lst = list()
    if not os.path.isdir(f"./output/{program}_{env}"):
        os.system(f"mkdir ./output/{program}_{env}")
    for n in range(N):
        bucket = storage.Client().bucket('kiran-research')
        blob = bucket.blob(f"latent-actions/outputs/{program}/seed{seed + n}/{env}/{item}.npy")
        blob.download_to_filename(f"./output/{env}/{program}/{seed + n}/{item}.npy")
        lst.append(np.load(f"./output/{env}/{program}/{seed + n}/{item}.npy"))
    print(f"from {program}: {[l.size for l in lst]}")
    arr = np.concatenate([l[np.newaxis, :] for l in lst], axis=0)
    return arr
    

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python run_gcp.py {N} {start_seed} {env} [{color} {program}] ...")
    else:
        N = 10
        seed = 1970
        item = "reward"
        programs = ["bcq", "ours", "td3_bc"]
        colors = ["blue", "orange", "green"]
        env = 'halfcheetah-expert-v0'
        plt.figure()
        for i in range(len(programs)):
            rewards = get_item_data(item, N, seed, programs[i], env)
            mean = rewards.mean(-1)
            std = rewards.std(-1)
            eval_freq = 5000
            max_timesteps = 1000000
            color = sys.argv[i]
            program = sys.argv[i + 1]
            x = range(eval_freq, eval_freq + max_timesteps, eval_freq)
            plt.plot(x, mean, color=color, label=program)
            plt.fill_between(x, mean - std, mean + std, facecolor=color, alpha=0.2)
        plt.legend()
        plt.show()