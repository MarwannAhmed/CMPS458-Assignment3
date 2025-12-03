import numpy as np
import torch
import torch.nn.functional as F

from models.actor import Actor
from models.critic import Critic

from utils.defs import Transition

# PPO Agent Implementation
class PPOAgent:

    def __init__(self, state_dim, action_dim, hidden_dim, gamma, actor_lr, critic_lr, entropy_coef, clip_epsilon=0.2, gae_lambda=0.95, value_coef=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.clip_epsilon = clip_epsilon

        self.memory = list()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()
    
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def compute_gae(self, rewards, values, next_values, dones):
        """
        Compute Generalized Advantage Estimation (GAE)
        """
        advantages = []
        gae = 0
        
        # Process in reverse order
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE: A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)
    
    def update(self, n_epochs=10, batch_size=64):
        """
        Update policy using PPO with multiple epochs and minibatches
        """
        if len(self.memory) == 0:
            return 0.0, 0.0
        
        # Extract batch data
        batch = Transition(*zip(*self.memory))
        states = torch.FloatTensor(np.array(batch.state))
        next_states = torch.FloatTensor(np.array(batch.next_state))
        actions = torch.LongTensor(np.array(batch.action))
        rewards = torch.FloatTensor(np.array(batch.reward))
        dones = torch.FloatTensor(np.array(batch.done))
        old_log_probs = torch.FloatTensor(np.array(batch.log_prob))

        # Compute values for current and next states
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            
            # Compute GAE advantages
            advantages = self.compute_gae(rewards, values, next_values, dones)
            
            # Compute returns (for value function target)
            returns = advantages + values
            
            # Normalize advantages (important for stable training)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store for logging
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_updates = 0
        
        # Multiple epochs of updates
        dataset_size = len(self.memory)
        
        for epoch in range(n_epochs):
            # Generate random minibatches
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Mini-batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Compute current log probs and entropy
                action_probs = self.actor(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute probability ratio
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                
                # Actor loss (negative because we want to maximize)
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # Gradient clipping
                self.actor_optimizer.step()
                
                # Critic loss (value function)
                batch_values = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(batch_values, batch_returns)
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # Gradient clipping
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_updates += 1
        
        avg_actor_loss = total_actor_loss / total_updates if total_updates > 0 else 0.0
        avg_critic_loss = total_critic_loss / total_updates if total_updates > 0 else 0.0
        
        return avg_actor_loss, avg_critic_loss
    
    def reset_memory(self):
        self.memory = list()
    
    def save_models(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    def load_models(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))