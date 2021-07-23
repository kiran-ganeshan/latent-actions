import os, shutil
if os.path.exists("/root/.mujoco/mjkey.txt"):
    shutil.copyfile("/code/mjkey.txt", "/root/.mujoco/mjkey.txt")
    print("Mujoco Setup Complete")
