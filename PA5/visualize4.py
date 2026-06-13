import numpy as np
import gymnasium as gym
import pygame
from RobotNavigation import RobotNavigationEnv
from stable_baselines3 import DQN   # If you trained a PPO model you have to import PPO here. I hope you get the gist.


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
        
        pass


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
    

# Load the trained model
model = DQN.load("MODEL4")   # If you used PPO the you have to use PPO.load("MODEL3"). I hope you get the gist.


# Initiate the robot navigation environment.
env = RobotNavigationEnv()
H = 20  # This MUST be same as the one used during training.
env = ModifiedRobotNavigationEnv(env, H)

# If you want to see an animation, then initiate the environment as follows:
#
# env = RobotNavigationEnv(render_mode='human')
#
# Animations are fun but also time consuming.


# Simulate the trained model for one episode
x, _ = env.reset()
terminated = False
truncated = False
total_reward = 0
while not(terminated) and not(truncated):
    a = model.predict(x, deterministic=True)
    x, r, terminated, truncated, _ = env.step(a[0])
    total_reward += r

print('Sum of reward = {}'.format(total_reward))

env.render() # This is used to generate the path/trail of the robot (no animation).

env.close()  # If you doing animation and you want to abruptly end it then
             # first stop the program execution and then run env.close().
