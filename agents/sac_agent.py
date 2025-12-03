import numpy as np
import torch

from models.actor import Actor
from models.critic import Critic

from utils.defs import Transition

# SAC Agent Implementation
class SACAgent:

    def __init__(self, state_dim, action_dim, hidden_dim, gamma, actor_lr, critic1_lr, critic2_lr, entropy_coef):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic1 = Critic(state_dim, hidden_dim)
        self.critic2 = Critic(state_dim, hidden_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic1_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic2_lr)

        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic1_lr = critic1_lr
        self.critic2_lr = critic2_lr
        self.entropy_coef = entropy_coef

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
            next_state_values_1 = self.critic1(next_state_batch)
            next_state_values_2 = self.critic2(next_state_batch)
            next_state_values = torch.min(next_state_values_1, next_state_values_2)
            targets = reward_batch + self.gamma * next_state_values * (1 - done_batch)
        
        # Update Critic 1
        state_values = self.critic1(state_batch)
        critic1_loss = torch.nn.functional.mse_loss(state_values, targets)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Update Critic 2
        state_values = self.critic2(state_batch)
        critic2_loss = torch.nn.functional.mse_loss(state_values, targets)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update Actor
        action_probs = self.actor(state_batch)
        selected_action_probs = action_probs.gather(1, action_batch)
        action_log_probs = torch.log(selected_action_probs + 1e-8)
        advantages = targets - state_values.detach()
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        actor_loss = -(action_log_probs * advantages).mean() - self.entropy_coef * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()

    def reset_memory(self):
        self.memory = list()
    
    def save_models(self, actor_path, critic1_path, critic2_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic1_path)
        torch.save(self.critic2.state_dict(), critic2_path)
    
    def load_models(self, actor_path, critic1_path, critic2_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic2.load_state_dict(torch.load(critic2_path))