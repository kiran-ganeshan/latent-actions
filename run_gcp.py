import doodad
from doodad import MountLocal, MountGCP, GCPMode
from doodad.utils import TESTING_DIR
from testing.config import GCP_PROJECT, GCP_BUCKET, GCP_IMAGE
from datetime import datetime
import os, sys
from time import sleep
            
'''
Runs script on GCP using doodad.
Usage: python run_gcp.py {num runs} {seed} {name} --flag value --flag value ...
Example: To run d4rl_evaluations/bcq 10 times on seed 100000 with flags {env: halfcheetah, tau: 0.5}:
python run_gcp.py 10 100000 bcq --env halfcheetah --tau 0.5
'''

def run(loc, seed, name, env, flags):
    repo = 'd4rl_evaluations'
    local_mount = MountLocal(
        local_dir=TESTING_DIR,
        mount_point='/data',
        output=False)
    gcp_mount = MountGCP(
        gcp_path=os.path.join(os.path.join(name, f'seed{seed}'), env),
        mount_point=f'/output')
    code_mount1 = MountLocal(
        local_dir=f'~/latent-actions/{repo}/{name}',
        mount_point='/code', pythonpath=True)
    code_mount2 = MountLocal(
        local_dir='~/.d4rl/datasets',
        mount_point='/d4rl')
    mounts = [local_mount, gcp_mount, code_mount1, code_mount2]
    redirects = '> /output/output.log 2> /output/error.log'
    flags = (" " if len(flags) != 0 else "") + " ".join(flags)
    cmd = f"python -c \"import d4rl; d4rl.set_dataset_path('/d4rl');\" {redirects};\n"
    cmd += f"cd /code {redirects};\npython main.py{flags} --seed {seed} --env {env} {redirects}"
    #cmd = f"pip list | grep mujoco-py {redirects};cd /code;\npython test.py {redirects}"
    print(f"running command:\n{cmd}")
    locations = ['us-west4-a', 'us-west1-a', 'us-east1-b', 'asia-east2-a', 
                 'asia-northeast1-a', 'asia-northeast2-a']
    launcher = GCPMode(
        gcp_bucket=GCP_BUCKET,
        gcp_log_path='latent-actions',
        gcp_project=GCP_PROJECT,
        instance_type='n2-standard-4',
        zone=locations[loc],
        gcp_image=GCP_IMAGE,
        gcp_image_project=GCP_PROJECT
    )
    finished = False
    while not finished:
        try:
            doodad.run_command(
                docker_image='ikostrikov/ml_cpu_new:latest',
                command=cmd,
                mode=launcher,
                mounts=mounts,
                verbose=True
            )
            finished = True
        except:
            print("connection failed. attempting to run again.")
            sleep(5.)
        

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print("Usage: python run_gcp.py {location} {num runs} {seed} {name} {env} --flag value ...")
    else:
        for i in range(int(sys.argv[2])):
            run(int(sys.argv[1]), str(int(sys.argv[3]) + i), sys.argv[4], sys.argv[5], sys.argv[6:])
