import gymnasium as gym
import numpy as np

class MountainCarRewardWrapper(gym.Wrapper):
    """
    Reward shaping for MountainCar to encourage:
    1. Getting to higher positions (potential energy)
    2. Having velocity in the right direction
    3. Reaching the goal
    """
    def __init__(self, env):
        super().__init__(env)
        self.best_position = -0.5  # Starting position
        self.position_threshold = 0.5  # Goal position
        
    def reset(self, **kwargs):
        self.best_position = -0.5
        return self.env.reset(**kwargs)
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        position = observation[0]
        velocity = observation[1]
        
        # Original reward is -1 per step
        
        # 1. Reward for reaching new rightmost position (height)
        position_reward = 0
        if position > self.best_position:
            position_reward = (position - self.best_position) * 100
            self.best_position = position
        
        # 2. Reward for velocity in the right direction when on the right side
        velocity_reward = 0
        if position > -0.5:  # Past the starting point
            velocity_reward = abs(velocity) * 10
        
        # 3. Big reward for reaching the goal
        goal_reward = 0
        if terminated and position >= self.position_threshold:
            goal_reward = 1000
        
        # Combine rewards
        shaped_reward = reward + position_reward + velocity_reward + goal_reward
        
        return observation, shaped_reward, terminated, truncated, info