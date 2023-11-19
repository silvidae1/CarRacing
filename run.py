import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv


model = PPO.load("tmp/best_model.zip")
env_id = "CarRacing-v2"
def main():
   env = gym.make(env_id, render_mode="human", domain_randomize=False, continuous=False)
   env = MaxAndSkipEnv(env, 4)

   obs = env.reset()
   terminated = False
   while not terminated:
       action = env.action_space.sample()
       obs, reward, terminated, truncated, info = env.step(action)
       env.render()

if __name__ == "__main__":
    main()