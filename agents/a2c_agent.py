import numpy as np
import torch

from models.actor import Actor
from models.critic import Critic

from utils.defs import Transition

# A2C Agent Implementation
class A2CAgent:

    def __init__(self, state_dim, action_dim, hidden_dim, gamma, actor_lr, critic_lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.memory = list()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        action = torch.multinomial(action_probs, 1).item()
        return action
    
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def update(self):
        batch = Transition(*zip(*self.memory))
        state_batch = torch.from_numpy(np.array(batch.state)).float()
        next_state_batch = torch.from_numpy(np.array(batch.next_state)).float()
        action_batch = torch.from_numpy(np.array(batch.action)).long().unsqueeze(1)
        reward_batch = torch.from_numpy(np.array(batch.reward)).float().unsqueeze(1)
        done_batch   = torch.from_numpy(np.array(batch.done)).float().unsqueeze(1)

        # Compute targets
        with torch.no_grad():
            next_state_values = self.critic(next_state_batch)
            targets = reward_batch + self.gamma * next_state_values * (1 - done_batch)
        
        # Update Critic
        state_values = self.critic(state_batch)
        critic_loss = torch.nn.functional.mse_loss(state_values, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        action_probs = self.actor(state_batch)
        selected_action_probs = action_probs.gather(1, action_batch)
        action_log_probs = torch.log(selected_action_probs + 1e-8)
        advantages = targets - state_values.detach()
        actor_loss = -(action_log_probs * advantages).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()
    
    def reset_memory(self):
        self.memory = list()
    
    def save_models(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    def load_models(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))