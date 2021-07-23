run_gcp: 
Runs script on GCP using doodad.
Usage: python run_gcp.py {num runs} {seed} {name} --flag value --flag value ...
Example: To run d4rl_evaluations/bcq 10 times on seed 100000 with flags {env: halfcheetah, tau: 0.5}:
python run_gcp.py 10 100000 bcq --env halfcheetah --tau 0.5
