from subprocess import run
import sys
run_exp = lambda *args, **kwargs: run(*args, stdout=sys.stdout, stderr=sys.stderr)
seed = 1990
envs = ["antmaze"]
datasets = ["umaze-v0", "umaze-diverse-v0", "medium-diverse-v0", "medium-play-v0", "large-diverse-v0", "large-play-v0"]
programs = ["bcq", "ours", "td3_bc", "td3_bc_vib"]
for env in envs:
    for dataset in datasets:
        for i, program in enumerate(programs):
            run_exp(f"python run_gcp.py {i} 10 {seed} {program} {env}-{dataset} --no_tqdm --load --save".split())