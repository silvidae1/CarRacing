import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from gymnasium.wrappers import FrameStack, ResizeObservation, GrayScaleObservation, RecordVideo
from stable_baselines3.common.atari_wrappers import (
    MaxAndSkipEnv,
    WarpFrame,
    NoopResetEnv
   )

import torch

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved. 
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

def make_env(env_id, rank, capture_video = False, seed = 0, run_name='Random'):
    def _init():
        env = gym.make(env_id, domain_randomize=False, continuous=False, render_mode='rgb_array')     # TODO: try with multidiscrete action

        if capture_video:
          if rank == 0:
            print("Video setup")
            env = RecordVideo(env, f"videos/{run_name}")

        # env = NoopResetEnv(env, noop_max = 30)
        env = MaxAndSkipEnv(env, 4)    # Only iterates the environment every 4 frames. Also fuses together last 2 to remove sprites and unique features
        # env = ResizeObservation(env, [84, 84])
        env = GrayScaleObservation(env, keep_dim=True)   # True for Cnn
        # env = FrameStack(env,num_stack=4, lz4_compress=False)   # cant run with cnn policy

        env.reset(seed=(seed+rank))
        return env

    set_random_seed(seed)
    return _init

# In 150k steps, we should already have around 400 reward
def main():
  # Do not change log directory
  run_name = "PPO_Cpu_Cnn_500k_0-0003_Grey"  # Try with 0.0003
  # run_name = "test"
  log_dir= "tmp/monitor/"+run_name ## SAME DIRECTORY FOR MONITOR AND BEST MODEL!!

  os.makedirs(log_dir, exist_ok=True) # Dirctory to save model
  os.makedirs(log_dir + run_name, exist_ok=True)  # Saves best model somewhere else

  env_id = "CarRacing-v2"
  num_cpu = 4

  env = VecMonitor(SubprocVecEnv([make_env(env_id, i, run_name=run_name, capture_video=True) for i in range(num_cpu)]), log_dir)
  # ent_coef in atari - 0.01
  model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0003, ent_coef=0.000,tensorboard_log="./board/")
  # model = PPO.load("tmp/best_model.zip", env=env, print_system_info=True)  
  # model.set_parameters(load_path_or_dict="tmp/best_model.zip")
  print("Observation Space: ", env.observation_space.shape)

  # #----------------------------- LEARNING -----------------------------------------------#
  print("Started Training")
  callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir = log_dir)
  model.learn(total_timesteps=500000, callback = callback, tb_log_name= run_name)  # O nome é algoritmo_Policy_timesteps_learning rate
  model.save(env_id)
  print("Finished Training")
  #----------------------------- Finished LEARNING -----------------------------------------------#
if __name__ == '__main__':
    main()