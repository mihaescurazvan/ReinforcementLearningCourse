import os
import subprocess

log_path = os.path.join('Training', 'Logs')
training_log_path = os.path.join(log_path, 'PPO_2')

subprocess.run("tensorboard --logdir=(training_log_path)")