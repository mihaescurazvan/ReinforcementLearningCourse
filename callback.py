import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import shimmy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


# load environment
environment_name = "CartPole-v1"
env = gym.make(environment_name, render_mode="human")
env = DummyVecEnv([lambda: env])

log_path = os.path.join('Training', 'Logs')

# adding a callback to the training stage
save_path = os.path.join('Training', 'Saved Models')
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(env,
                            callback_on_new_best=stop_callback,
                            eval_freq=10000,
                            best_model_save_path=save_path,
                            verbose=1)

# new PPO model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# learning with callback
model.learn(total_timesteps=20000, callback=eval_callback)

