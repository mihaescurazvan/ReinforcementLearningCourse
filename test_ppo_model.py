import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import shimmy

# load environment
environment_name = "CartPole-v1"
env = gym.make(environment_name, render_mode="human")
env = DummyVecEnv([lambda: env])

log_path = os.path.join('Training', 'Logs')


# load saved model
PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')
model = PPO.load(PPO_Path, env=env)

# testing

episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        score += reward
    print('Episode: {} Score: {}'.format(episode, score))

env.close()


