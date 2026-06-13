import numpy as np
import gymnasium as gym
import pygame
# Write just ONE line of code below this comment to import DQN from stable baseline
from stable_baselines3 import DQN


def visualize_model_performance(model):
    env = gym.make('MountainCar-v0', render_mode='human')
    
    x, _ = env.reset()
    total_reward = 0
    terminated, truncated = False, False
    while not(terminated) and not(truncated):
        action, _ = model.predict(x)
        x, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    
    print('Total reward = {}'.format(total_reward))
    env.close()
    # pygame.display.quit() # Use this line when the display screen is not going away
        

class Custom_Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Modified reward: weighted combination of 3 components
        # Component 1: position-based reward (higher = closer to goal at +0.5)
        position = observation[0]
        velocity = observation[1]
        
        position_reward = 5.0 * (position + 0.5)          # encourage moving right toward +0.5
        speed_reward = 10.0 * abs(velocity)                # encourage gaining momentum/speed
        step_penalty = -1.0                                 # penalty for every step not at goal
        
        modified_reward = position_reward + speed_reward + step_penalty
        
        return observation, modified_reward, terminated, truncated, info


# Initiate the mountain car environment.
env = gym.make('MountainCar-v0')

# Write just one line of code below this comment to create a modified environment using Custom_Wrapper class.
env = Custom_Wrapper(env)

# Write just TWO lines of code below this comment to train a DQN model for mountain car.
model = DQN('MlpPolicy', env, verbose=1, learning_rate=5e-4, buffer_size=50000,
            batch_size=64, train_freq=4, target_update_interval=500,
            exploration_initial_eps=1.0, exploration_final_eps=0.05,
            learning_starts=1000,
            policy_kwargs=dict(net_arch=[64, 64]))
model.learn(total_timesteps=200000)

# Close the mountain car environment.
env.close

# Write just ONE line of code below to save the DQN model that you have trained.
# YOU HAVE TO SUBMIT THIS MODEL. THE NAME OF THE MODEL MUST BE MODEL2.
# THE MODEL THAT YOU SUBMIT MUST NOT EXCEED 1 MB. ELSE ZERO FOR SECTION 4.
model.save("MODEL2")

# Write just ONE line of code below this comment to call visualize_model_performance in order to test the performance of the trained model
visualize_model_performance(model)
