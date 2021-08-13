from subprocess import run
import sys
run_exp = lambda *args, **kwargs: run(*args, stdout=sys.stdout, stderr=sys.stderr)
seed = 1990
loc = 3
envs = ["antmaze"]
datasets = ["umaze-v0", "umaze-diverse-v0", "medium-diverse-v0", "medium-play-v0", "large-diverse-v0", "large-play-v0"]
for env in envs:
    for dataset in datasets:
        run_exp(f"python run_gcp.py {loc} 10 {seed} td3_bc_vib {env}-{dataset}".split())