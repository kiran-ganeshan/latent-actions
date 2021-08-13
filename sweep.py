from subprocess import run
import sys
run_exp = lambda *args, **kwargs: run(*args, stdout=sys.stdout, stderr=sys.stderr)
seed = 1990
loc = 0
betas = [0.01, 0.1, 1, 10, 100]
envs = ["halfcheetah", "walker2d"]
datasets = ["random-v0", "medium-v0", "expert-v0", "medium-replay-v0", "medium-expert-v0"]
for env in envs:
    for dataset in datasets:
        for i, beta in enumerate(betas):
            run_exp(f"python run_gcp.py {loc} 1 {seed + i} ours {env}-{dataset} --beta {beta}".split())
loc = 1
envs = ["hopper", "ant"]
for env in envs:
    for dataset in datasets:
        for i, beta in enumerate(betas):
            run_exp(f"python run_gcp.py {loc} 1 {seed + i} ours {env}-{dataset} --beta {beta}".split())
loc = 2
envs = ["antmaze"]
datasets = ["umaze-v0", "umaze-diverse-v0", "medium-diverse-v0", "medium-play-v0", "large-diverse-v0", "large-play-v0"]
for env in envs:
    for dataset in datasets:
        run_exp(f"python run_gcp.py {loc} 10 {seed} td3_bc {env}-{dataset}".split())
loc = 3
for env in envs:
    for dataset in datasets:
        run_exp(f"python run_gcp.py {loc} 10 {seed} td3_bc_vib {env}-{dataset}".split())