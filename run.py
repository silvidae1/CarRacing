import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation


model = PPO.load("tmp\monitor\PPO_Cnn_600k_0-0003_Grey_ent_0_01/best_model.zip")
env_id = "CarRacing-v2"
def main():
    env = gym.make(env_id, render_mode="human", domain_randomize=False, continuous=False)
    env = MaxAndSkipEnv(env, 4)
    # env = ResizeObservation(env, [84, 84])
    env = GrayScaleObservation(env, keep_dim=True)

    obs = env.reset()
    obs = obs[0]
    terminated = False

    print("-------- Running ---------")
    while not terminated:
        action, state = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

if __name__ == "__main__":
    main()

