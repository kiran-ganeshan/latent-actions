import doodad
from doodad import MountLocal, MountGCP, GCPMode
from doodad.utils import TESTING_DIR
from testing.config import GCP_PROJECT, GCP_BUCKET, GCP_IMAGE
import os, sys
            
'''
Runs script on GCP using doodad.
Usage: python run_gcp.py {num runs} {seed} {name} --flag value --flag value ...
Example: To run d4rl_evaluations/bcq 10 times on seed 100000 with flags {env: halfcheetah, tau: 0.5}:
python run_gcp.py 10 100000 bcq --env halfcheetah --tau 0.5
'''

def run(N, seed, name, flags):
    
    local_mount = MountLocal(
        local_dir=TESTING_DIR,
        mount_point='/data',
        output=False)
    gcp_mount = MountGCP(
        gcp_path=os.path.join(name, f'seed{seed}'),
        mount_point=f'/output')
    code_mount1 = MountLocal(
        local_dir='~/latent-actions',
        mount_point='/code', pythonpath=True)
    code_mount2 = MountLocal(
        local_dir='~/latent-actions/doodad',
        mount_point='/code/doodad', pythonpath=True)
    mounts = [local_mount, gcp_mount, code_mount1, code_mount2]
    repo = '/code/d4rl_evaluations/'
    flags = (" " if len(flags) != 0 else "") + " ".join(flags)
    cmd = f"cd /code;\npython move_mjkey.py;\ncd {repo}{name};\n"
    for s in range(int(seed), int(seed) + int(N)):
        cmd += f"python main.py{flags} --seed {s} >> /outputoutput.log 2> /output/error.log;\n"
    #cmd = "source ~/.bashrc; which conda >> /output/output.log"
    print("running command:\n{}".format(cmd))
    launcher = GCPMode(
        gcp_bucket=GCP_BUCKET,
        gcp_log_path='latent-actions',
        gcp_project=GCP_PROJECT,
        instance_type='e2-medium',
        zone='us-west1-a',
        gcp_image=GCP_IMAGE,
        gcp_image_project=GCP_PROJECT
    )
    doodad.run_command(
        docker_image='ikostrikov/ml_cpu:latest',
        command=cmd,
        mode=launcher,
        mounts=mounts,
        verbose=True
    )

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
