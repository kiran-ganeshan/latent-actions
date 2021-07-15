"""
Instructions:
1) Set up testing/config.py (copy from config.py.example and fill in the fields)
2) Run this script
3) Look inside your AZ_CONTAINER and you should see results in test_azure_with_mounts/azure_script_output/output.out
"""
import doodad
from doodad.utils import TESTING_DIR
from testing.config import GCP_PROJECT, GCP_BUCKET, GCP_IMAGE
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('sweep_id', None, 'The ID of the sweep for which to start an agent.')

def run():
    local_mount = doodad.MountLocal(
        local_dir=TESTING_DIR,
        mount_point='/data',
        output=False
    )
    code_mount = doodad.MountLocal(
        local_dir='/Users/kiranganeshan/latent-actions/density_models',
        mount_point='/code/density_models', 
        pythonpath=True
    )
    doodad_mount = doodad.MountLocal(
        local_dir='/Users/kiranganeshan/latent-actions/doodad',
        mount_point='/code/doodad', 
        pythonpath=True
    )
    gcp_mount = doodad.MountGCP(
        gcp_path='secret_output',
        mount_point='/output'
    )
    mounts = [gcp_mount, local_mount, code_mount, doodad_mount]
    launcher = doodad.GCPMode(
        gcp_bucket=GCP_BUCKET,
        gcp_log_path='test_doodad/gcp_gpu_test',
        gcp_project=GCP_PROJECT,
        instance_type='n1-standard-1',
        zone='us-west1-a',
        gcp_image=GCP_IMAGE,
        gcp_image_project=GCP_PROJECT,
        use_gpu=True,
        gpu_model='nvidia-tesla-t4'
    )
    doodad.run_command(
        command='wandb agent kbganeshan/latent-actions/{}'.format(FLAGS.agent_str),
        mode=launcher,
        mounts=mounts,
        verbose=True
    )
    
if __name__ == '__main__':
    run()
