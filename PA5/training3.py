import numpy as np
import gymnasium as gym
import pygame
from RobotNavigation import RobotNavigationEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
# Imported the libraries which I think are useful. You can import the rest.


class LoggingAndSavingCallback(BaseCallback):
    def __init__(self, test_period, test_count, verbose=0):
        super().__init__(verbose)
        # test_period is the number of time steps (env.step()) after which we
        # want to test the model. You also have to save the latest model every
        # test_period.
        
        # test_count is the number of episodes for which we want to test the model.
        
        self.test_period = test_period
        self.test_count = test_count
        
        # Track current episode reward sum
        self.current_episode_reward = 0.0
        
        # List to store sum of reward per episode during training
        self.training_rewards = []
        
        # Best average test reward seen so far
        self.best_avg_test_reward = -np.inf
        
        # List to store average test rewards
        self.testing_rewards = []

    def _on_step(self) -> bool:
        
        # Accumulate reward for current episode
        self.current_episode_reward += self.locals['rewards'][0]
        
        # 1. At end of each episode, save training_log.npy
        if self.locals['dones'][0]:
            self.training_rewards.append(self.current_episode_reward)
            np.save('training_log.npy', np.array(self.training_rewards))
            self.current_episode_reward = 0.0
        
        # 2. Every test_period steps: save latest model, test, update best model
        if self.num_timesteps % self.test_period == 0:
            # 2.a. Save the latest model
            self.model.save("LATEST_MODEL")
            
            # 2.b. Test the latest model over test_count episodes
            test_env = RobotNavigationEnv()
            episode_rewards = []
            
            for _ in range(self.test_count):
                obs, _ = test_env.reset()
                ep_reward = 0.0
                terminated = False
                truncated = False
                while not terminated and not truncated:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = test_env.step(action)
                    ep_reward += reward
                episode_rewards.append(ep_reward)
            
            test_env.close()
            
            avg_reward = np.mean(episode_rewards)
            self.testing_rewards.append(avg_reward)
            np.save('testing_log.npy', np.array(self.testing_rewards))
            
            # Save as best model if this is the highest average reward so far
            if avg_reward > self.best_avg_test_reward:
                self.best_avg_test_reward = avg_reward
                self.model.save("BEST_MODEL")
            
            if self.verbose > 0:
                print(f"Timestep {self.num_timesteps}: avg test reward = {avg_reward:.2f}")
            
        return True # This MUST return True unless you want the training to stop.


# Initiate the robot navigation environment.
env = RobotNavigationEnv()


# Initiate an instance of the LoggingAndSavingCallback. Desription of test_period
# and test_count are there in _init__ function of LoggingAndSavingCallback.
test_period = 20000   # Default value. You can change it.
test_count = 10       # Default value. You can change it.
callback = LoggingAndSavingCallback(test_period, test_count)


# The code that you use to train the RL agent for the robot navigation environment
# goes below this line. The total number of lines is unlikely to be more than 10.
model = DQN(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100000,
    batch_size=64,
    train_freq=4,
    target_update_interval=1000,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    exploration_fraction=0.3,
    learning_starts=5000,
    policy_kwargs=dict(net_arch=[256, 256])
)
model.learn(total_timesteps=2000000, callback=callback)


# Close the robot navigation environment.
env.close


# Write just ONE line of code below to save the model that you have trained.
# YOU HAVE TO SUBMIT THIS MODEL. THE NAME OF THE MODEL MUST BE MODEL3.
model.save("MODEL3")
