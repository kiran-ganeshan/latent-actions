from subprocess import run
import sys
run_exp = lambda *args, **kwargs: run(*args, stdout=sys.stdout, stderr=sys.stderr)
seed = 1980
envs = ["hopper", "ant"]
datasets = ["random-v0", "medium-v0", "expert-v0", "medium-replay-v0", "medium-expert-v0"]
betas = [0.01, 0.1, 1, 10, 100]
for env in envs:
    for dataset in datasets:
        for i, beta in enumerate(betas):
            run_exp(f"python run_gcp.py 1 {seed + i} ours {env}-{dataset} --beta {beta}".split())