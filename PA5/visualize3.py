import numpy as np
import gymnasium as gym
import pygame
from RobotNavigation import RobotNavigationEnv
from stable_baselines3 import DQN   # If you trained a PPO model you have to import PPO here. I hope you get the gist.


# Load the trained model
model = DQN.load("MODEL3")   # If you used PPO the you have to use PPO.load("MODEL3"). I hope you get the gist.


# Initiate the robot navigation environment.
env = RobotNavigationEnv()

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
