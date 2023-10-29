import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import shimmy


# load environment
environment_name = "CartPole-v1"
env = gym.make(environment_name, render_mode="rgb_array")

episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, _, info = env.step(action)

        score += reward
    print('Episode: {} Score: {}'.format(episode, score))

env.close()

# train RL model

log_path = os.path.join('Training', 'Logs')
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=40000)

# save model
PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')
model.save(PPO_Path)
del model
model = PPO.load(PPO_Path, env=env)


