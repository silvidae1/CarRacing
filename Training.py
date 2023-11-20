import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

import os

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

def make_env(env_id, rank, seed = 0):
    def _init():
        env = gym.make(env_id, domain_randomize=False, continuous=False)
        env = MaxAndSkipEnv(env, 4)
        env.reset(seed=(seed+rank))
        return env

    set_random_seed(seed)
    return _init


def main():
  log_dir="tmp/" # Directory para o log

  os.makedirs(log_dir, exist_ok=True) #Se não existir cria

  env_id = "CarRacing-v2"
  num_cpu = 4

  env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]), "tmp/monitor")


  #env = gym.make("CarRacing-v2", render_mode="human", domain_randomize=False, continuous=False)

  model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./board/")
  #model = PPO.load("", env=env)


  # #----------------------------- LEARNING -----------------------------------------------#
  print("Started Training")
  callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir = log_dir)
  model.learn(total_timesteps=500000, callback =callback, tb_log_name= "PPO_Cnn_100_deault")  # O nome é algoritmo_Policy_timesteps_learning rate
  model.save(env_id)
  print("Finished Training")
  #----------------------------- Finished LEARNING -----------------------------------------------#
if __name__ == '__main__':
    main()