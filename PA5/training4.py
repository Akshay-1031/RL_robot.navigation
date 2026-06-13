import numpy as np
import gymnasium as gym
import pygame
from RobotNavigation import RobotNavigationEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


class ModifiedRobotNavigationEnv(gym.Wrapper):
    def __init__(self, env, H):
        super().__init__(env)
        self.H = H   # Number of time slots in a mini episode.
        
        # The action space of the modified robot navigation environment.
        Delta = H*self.env.delta
        self.action_space = gym.spaces.Box(-Delta*np.ones(2), Delta*np.ones(2), dtype=np.float32)
        
    
    def conventional_policy(self, robot_position, goal_intermediate):
        # For don't move, return -1.
        # For up, return 0.
        # For down, return 1.
        # For left, return 2.
        # For right, return 3.
        
        # FREE ADVICE: To account for decimal error, it is better to give some
        # level to tolerance when comparing two values. Examples:
        # 1) Rather then writting if a>b, it is better to write if a-b>0.00001.
        # 2) Rather then writting if a<b, it is better to write if a-b<-0.00001.
        
        x_tilde = robot_position[0]
        y_tilde = robot_position[1]
        x_goal = goal_intermediate[0]
        y_goal = goal_intermediate[1]
        
        if x_goal - x_tilde > 0.00001:        # goal is to the right
            return 3
        elif x_goal - x_tilde < -0.00001:     # goal is to the left
            return 2
        elif y_goal - y_tilde > 0.00001:      # goal is above
            return 0
        elif y_goal - y_tilde < -0.00001:     # goal is below
            return 1
        else:                                  # already at intermediate goal
            return -1


    def step(self, action):
        # Compute the intermediate goal
        goal_intermediate = self.env.robot_position + action        
        grid_x = int(goal_intermediate[0]/self.env.delta) + 1
        grid_y = int(goal_intermediate[1]/self.env.delta) + 1        
        grid_x = self.env.delta*(0.5 + (grid_x - 1))
        grid_y = self.env.delta*(0.5 + (grid_y - 1))
        goal_intermediate = np.array([grid_x, grid_y])
        
        # Simulate a mini-episode using the conventional policy.
        reward_miniepisode = 0
        for h in range(self.H):
            reward = -np.sqrt(np.sum((self.env.robot_position - self.env.goal)**2))
            
            a = self.conventional_policy(self.env.robot_position, goal_intermediate)  # Call conventional policy.
            
            if a!=-1: # If a==-1, then robot position does not change.
                self.env.robot_position = self.env.robot_position + self.env.action_dict[a+0]*self.env.delta
            
            self.env.trail.append(self.env.robot_position)
            
            # Check for collision
            terminated = self.env.check_collision()
            if terminated:
                reward = -10000
            
            # Check if goal is reached
            if not(terminated):
                if (self.env.goal[0] - self.env.robot_position[0])**2 + (self.env.goal[1] - self.env.robot_position[1])**2<=self.env.goal_radius**2:
                    terminated = True
                    reward = 1000
            
            reward_miniepisode+=reward
            
            self.env.t += 1
            truncated = False
            if self.env.t>self.env.Horizon:
                truncated = True
            
            if self.env.render_mode == "human":
                self.env.render()
            
            if terminated or truncated:
                break
        
        self.env.observation = np.concatenate((self.env.get_lidar_reading(),self.env.robot_position))
        
        return self.env.observation, reward_miniepisode, terminated, truncated, {}
        
  

# You can copy-paste the code for the custom callback LoggingAndSavingCallback
# that you wrote for training3.py. All you need to change is the code for
# initiating the environment during testing.
class LoggingAndSavingCallback(BaseCallback):
    def __init__(self, test_period, test_count, verbose=0):
        super().__init__(verbose)
        
        self.test_period = test_period
        self.test_count = test_count
        
        self.current_episode_reward = 0.0
        self.training_rewards = []
        self.best_avg_test_reward = -np.inf
        self.testing_rewards = []
        
    def _on_step(self) -> bool:
        
        # Accumulate reward for current episode
        self.current_episode_reward += self.locals['rewards'][0]
        
        # 1. At end of each episode, save training_log.npy
        if self.locals['dones'][0]:
            self.training_rewards.append(self.current_episode_reward)
            np.save('training_log4.npy', np.array(self.training_rewards))
            self.current_episode_reward = 0.0
        
        # 2. Every test_period steps
        if self.num_timesteps % self.test_period == 0:
            # 2.a. Save the latest model
            self.model.save("LATEST_MODEL4")
            
            # 2.b. Test the latest model using a LOCAL modified environment
            H = 20
            test_base_env = RobotNavigationEnv()
            test_env = ModifiedRobotNavigationEnv(test_base_env, H)
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
            np.save('testing_log4.npy', np.array(self.testing_rewards))
            
            if avg_reward > self.best_avg_test_reward:
                self.best_avg_test_reward = avg_reward
                self.model.save("BEST_MODEL4")
            
            if self.verbose > 0:
                print(f"Timestep {self.num_timesteps}: avg test reward = {avg_reward:.2f}")
        
        return True
            
            
# Initiate the robot navigation environment. 
env = RobotNavigationEnv()
H = 20  # H is the duration of a mini-episode. My advide is to change it between 10 to 40. But you can go crazy with it!
env = ModifiedRobotNavigationEnv(env, H)


# Initiate an instance of the LoggingAndSavingCallback. Desription of test_period
# and test_count are there in _init__ function of LoggingAndSavingCallback.
test_period = 20000   # Default value. You can change it.
test_count = 10       # Default value. You can change it.
callback = LoggingAndSavingCallback(test_period, test_count)


# The code that you use to train the RL agent for the robot navigation environment
# goes below this line. The total number of lines is unlikely to be more than 10.
model = SAC(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    learning_starts=5000,
    policy_kwargs=dict(net_arch=[256, 256])
)
model.learn(total_timesteps=2000000, callback=callback)


# Close the robot navigation environment.
env.close


# Write just ONE line of code below to save the model that you have trained.
# YOU HAVE TO SUBMIT THIS MODEL. THE NAME OF THE MODEL MUST BE MODEL4.
model.save("MODEL4")
