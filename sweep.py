from subprocess import run
import sys
run_exp = lambda *args, **kwargs: run(*args, stdout=sys.stdout, stderr=sys.stderr)
seed = 1990
envs = ["antmaze"]
datasets = ["umaze-v0"]#, "umaze-diverse-v0", "medium-diverse-v0", "medium-play-v0", "large-diverse-v0", "large-play-v0"]
programs = ["td3_bc", "td3_bc_vib", "ours", "bcq"]
locs = [0, 1, 2, 3, 4, 5]
for env in envs:
    for dataset in datasets:
        for i, program in enumerate(programs):
            run_exp(f"python run_gcp.py {locs[i]} 1 {seed} {program} {env}-{dataset}".split())