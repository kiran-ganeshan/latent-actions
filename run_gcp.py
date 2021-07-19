"""
Instructions:
1) Set up testing/config.py (copy from config.py.example and fill in the fields)
2) Run this script
3) Look inside your GCP_BUCKET under test_doodad and you should see results in secret.txt
"""
import doodad
from doodad.utils import TESTING_DIR
from testing.config import GCP_PROJECT, GCP_BUCKET, GCP_IMAGE
from doodad.wrappers.sweeper import launcher
            

def run():
    local_mount = doodad.MountLocal(
        local_dir=TESTING_DIR,
        mount_point='/data',
        output=False)
    gcp_mount = doodad.MountGCP(
        gcp_path='sweep_output',
        mount_point='/output')
    code_mount1 = doodad.MountLocal(
        local_dir='~/latent-actions',
        mount_point='/home/kg23/latent-actions', pythonpath=True)
    code_mount2 = doodad.MountLocal(
        local_dir='~/latent-actions/doodad',
        mount_point='/home/kg23/latent-actions/doodad', pythonpath=True)
    mounts = [local_mount, gcp_mount, code_mount1, code_mount2]
    launcher = doodad.GCPMode(
        gcp_bucket=GCP_BUCKET,
        gcp_log_path='test_doodad/gcp_test',
        gcp_project=GCP_PROJECT,
        instance_type='f1-micro',
        zone='us-west1-a',
        gcp_image=GCP_IMAGE,
        gcp_image_project=GCP_PROJECT
    )
    doodad.run_command(
        command="python BCQ/continuous_BCQ/main.py \
                    --env Hopper-v3 \
                    --seed 10 \
                    --eval_freq 5e3 \
                    --max_timesteps 1e6 \
                    --batch_size 60 \
                    --discount 0.99 \
                    --tau 0.005 \
                    --lmbda 0.75 \
                    --phi 0.05",
        mode=launcher,
        mounts=mounts,
        verbose=True
    )

if __name__ == '__main__':
    run()
