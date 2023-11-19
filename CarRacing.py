import gymnasium as gym
#from gymnasium.wrappers.record_video import RecordVideo
from stable_baselines3 import PPO

print("olá")
env = gym.make("CarRacing-v2", render_mode="rgb_array", domain_randomize=False, continuous=False)

#model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=100)
#model.save("ppo_cartpole")

#del model # remove to demonstrate saving and loading

#model = PPO.load("ppo_cartpole")
print("olá 2")
obs = env.reset()
terminated = False

i=0
while i <500:
    action = env.action_space.sample()
    i+=1
    print(i)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    #env.monitor.start('results/carrace-1')

env.close()