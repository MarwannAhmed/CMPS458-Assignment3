import os
from collections import deque

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt

from agents.ppo_agent import PPOAgent
from config import PPO_CONFIG, ENVIRONMENTS
from utils.defs import Transition
from utils.discrete_pendulum import DiscretePendulum
from utils.mountain_car_wrapper import MountainCarRewardWrapper
import wandb


# ------------------------------------------------------------------
# Logging configuration: all output goes to output.log (no terminal)
# ------------------------------------------------------------------
def log(message):
    with open("output.log", "a") as f:
        f.write(message + "\n")
        f.flush()


# ------------------------------------------------------------------
# Utility: create environment (with discrete wrapper for Pendulum)
# ------------------------------------------------------------------
def create_env(env_name: str, render_mode=None):
    """
    Create an environment. For Pendulum-v1, wrap it with DiscretePendulum.
    """
    if env_name == "Pendulum-v1":
        base_env = gym.make(env_name, render_mode=render_mode)
        env = DiscretePendulum(base_env, num_actions=5)
    elif env_name == "MountainCar-v0":
        base_env = gym.make(env_name, render_mode=render_mode)
        env = MountainCarRewardWrapper(base_env)
    else:
        env = gym.make(env_name, render_mode=render_mode)
    return env


def create_video_env(env_name: str, video_dir: str):
    """
    Create an environment wrapped with RecordVideo to save videos.
    Uses rgb_array rendering.
    """
    os.makedirs(video_dir, exist_ok=True)

    if env_name == "Pendulum-v1":
        base_env = gym.make(env_name, render_mode="rgb_array")
        wrapped = DiscretePendulum(base_env, num_actions=5)
        env = RecordVideo(
            wrapped,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: True,
            name_prefix=f"PPO-{env_name}",
        )
    elif env_name == "MountainCar-v0":
        base_env = gym.make(env_name, render_mode="rgb_array")
        wrapped = MountainCarRewardWrapper(base_env)
        env = RecordVideo(
            wrapped,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: True,
            name_prefix=f"PPO-{env_name}",
        )
    else:
        base_env = gym.make(env_name, render_mode="rgb_array")
        env = RecordVideo(
            base_env,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: True,
            name_prefix=f"PPO-{env_name}",
        )
    return env


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
# In the train_agent_on_env function, replace the training loop:

def train_agent_on_env(
    env_name: str,
    max_episodes: int,
    target_reward: float,
    config: dict,
    rollout_length: int = 2048
):
    """
    Train a PPO agent on a given environment.
    """
    log(f"========== Training on environment: {env_name} ==========")

    env = create_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    gamma = config["Gamma"]
    actor_lr = config["Actor LR"]
    critic_lr = config["Critic LR"]
    entropy_coef = config["Entropy Coef"]
    hidden_dim = config["Hidden Dim"]

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        gamma=gamma,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        entropy_coef=entropy_coef,
    )

    run = wandb.init(project="cmps458_assignment3", name=f"PPO-{env_name}", config={**config, "env_id": env_name, "agent": "PPO"})

    episode_returns = []
    episode_durations = []
    moving_avg_returns = []
    actor_losses_per_update = []
    critic_losses_per_update = []
    recent_returns = deque(maxlen=100)

    episode = 0
    total_steps = 0
    
    # Track latest losses for logging
    latest_actor_loss = None
    latest_critic_loss = None
    
    state, _ = env.reset()
    episode_return = 0.0
    episode_length = 0
    
    while episode < max_episodes:
        # Collect rollout
        agent.reset_memory()
        
        for _ in range(rollout_length):
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
            total_steps += 1
            
            transition = Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=float(done),
                log_prob=log_prob
            )
            agent.store_transition(transition)
            
            state = next_state
            
            if done:
                episode += 1
                episode_returns.append(episode_return)
                episode_durations.append(episode_length)
                recent_returns.append(episode_return)
                
                avg_return_last_100 = np.mean(recent_returns)
                moving_avg_returns.append(avg_return_last_100)
                
                log(
                    f"[TRAIN] Env: {env_name} | Episode: {episode} | "
                    f"Return: {episode_return:.2f} | "
                    f"Duration: {episode_length} | "
                    f"AvgReturn(100): {avg_return_last_100:.2f}"
                )
                
                # Log to wandb with losses if available
                log_dict = {
                    "episode": episode,
                    "return": episode_return,
                    "avg100": avg_return_last_100,
                    "duration": episode_length,
                    "steps": total_steps
                }
                
                # Add losses if we've done at least one update
                if latest_actor_loss is not None:
                    log_dict["actor_loss"] = latest_actor_loss
                if latest_critic_loss is not None:
                    log_dict["critic_loss"] = latest_critic_loss
                
                wandb.log(log_dict)
                
                state, _ = env.reset()
                episode_return = 0.0
                episode_length = 0
                
                if episode >= max_episodes:
                    break
        
        # Update policy after collecting rollout
        if len(agent.memory) > 0:
            actor_loss, critic_loss = agent.update(n_epochs=10, batch_size=64)
            actor_losses_per_update.append(actor_loss)
            critic_losses_per_update.append(critic_loss)
            
            # Store latest losses for next episode logs
            latest_actor_loss = actor_loss
            latest_critic_loss = critic_loss
            
            log(
                f"[UPDATE] ActorLoss: {actor_loss:.6f} | CriticLoss: {critic_loss:.6f}"
            )
            
            # Also log the update itself
            wandb.log({
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "steps": total_steps
            })

    env.close()

    training_stats = {
        "episode_returns": episode_returns,
        "episode_durations": episode_durations,
        "moving_avg_returns": moving_avg_returns,
        "actor_losses": actor_losses_per_update,
        "critic_losses": critic_losses_per_update,
    }

    return agent, training_stats


# ------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------
def test_agent_on_env(
    env_name: str,
    agent: PPOAgent,
    num_episodes: int = 100,
):
    """
    Test a trained agent on the environment with greedy policy.
    Returns: test_durations, test_returns, avg_return
    """
    log(
        f"========== Testing agent on environment: {env_name} for {num_episodes} episodes =========="
    )

    env = create_env(env_name)
    test_durations = []
    test_returns = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        t = 0
        total_return = 0.0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = agent.actor(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward
            state = next_state
            t += 1

        test_returns.append(total_return)
        test_durations.append(t)

        log(
            f"[TEST] Env: {env_name} | Episode: {episode} | "
            f"Return: {total_return:.2f} | Duration: {t}"
        )

    env.close()

    avg_return = float(np.mean(test_returns))
    log(
        f"[TEST] Env: {env_name} | "
        f"Average Return over {num_episodes} episodes: {avg_return:.2f}"
    )

    return test_durations, test_returns, avg_return


# ------------------------------------------------------------------
# Plotting helper
# ------------------------------------------------------------------
def save_test_plots(env_name: str, test_durations, test_returns, figures_root="figures/PPO"):
    """
    Generate and save:
    - Episode duration vs episode
    - Episode return vs episode
    in figures/PPO/<env_name>
    """
    env_fig_dir = os.path.join(figures_root, env_name)
    os.makedirs(env_fig_dir, exist_ok=True)

    episodes = np.arange(1, len(test_returns) + 1)

    # 1) Episode duration vs Episode
    plt.figure()
    plt.plot(episodes, test_durations)
    plt.xlabel("Episode")
    plt.ylabel("Episode Duration (steps)")
    plt.title(f"PPO - {env_name} - Episode Duration vs Episode")
    plt.grid(True)
    duration_path = os.path.join(env_fig_dir, "episode_duration_vs_episode.png")
    plt.savefig(duration_path)
    plt.close()

    # 2) Episode return vs Episode
    plt.figure()
    plt.plot(episodes, test_returns)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title(f"PPO - {env_name} - Episode Return vs Episode")
    plt.grid(True)
    return_path = os.path.join(env_fig_dir, "episode_return_vs_episode.png")
    plt.savefig(return_path)
    plt.close()

    log(
        f"[PLOTS] Saved test plots for {env_name} to {env_fig_dir}"
    )


# ------------------------------------------------------------------
# Video recording helper
# ------------------------------------------------------------------
def record_agent_videos(
    env_name: str,
    agent: PPOAgent,
    num_episodes: int = 5,
    videos_root: str = "videos/PPO",
):
    """
    Record num_episodes episodes of a trained agent and save videos into
    videos/PPO/<env_name>.
    """
    env_video_dir = os.path.join(videos_root, env_name)
    env = create_video_env(env_name, env_video_dir)

    log(
        f"[VIDEO] Recording {num_episodes} episodes for {env_name} "
        f"into {env_video_dir}"
    )

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        t = 0
        total_return = 0.0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = agent.actor(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward
            state = next_state
            t += 1

        log(
            f"[VIDEO] Env: {env_name} | Recorded Episode: {episode} | "
            f"Return: {total_return:.2f} | Duration: {t}"
        )

    env.close()
    log(f"[VIDEO] Finished recording for {env_name}")


# ------------------------------------------------------------------
# Main loop over environments
# ------------------------------------------------------------------
def main():
    # Clear previous log
    if os.path.exists("output.log"):
        os.remove("output.log")
    
    # Iterate over all environments defined in ENVIRONMENTS / PPO_CONFIG
    for env_name, env_info in ENVIRONMENTS.items():
        if os.path.exists("trained_models/PPO/" + env_name + "/actor.pth") and os.path.exists("trained_models/PPO/" + env_name + "/critic.pth"):
            log(
                f"Trained models for {env_name} already exist. Skipping training."
            )
            continue
        if env_name not in PPO_CONFIG:
            log(
                f"Environment {env_name} has no PPO_CONFIG entry. Skipping."
            )
            continue

        max_episodes = PPO_CONFIG[env_name]["Training Episodes"]
        target_reward = env_info["target_reward"]
        config = PPO_CONFIG[env_name]

        log(
            f"\n\n==================== ENVIRONMENT: {env_name} ====================\n"
        )
        log(
            f"Max Episodes: {max_episodes} | Target Reward: {target_reward} | "
            f"Gamma: {config['Gamma']} | Actor LR: {config['Actor LR']} | "
            f"Critic LR: {config['Critic LR']} | Entropy Coef: {config['Entropy Coef']}"
        )

        # 2.a. Train the agent
        # In main(), update the training call:
        agent, training_stats = train_agent_on_env(
            env_name=env_name,
            max_episodes=max_episodes,
            target_reward=target_reward,
            config=config,
            rollout_length=config.get("Rollout Length", 2048),
        )

        # 2.b. Test the agent (100 episodes)
        test_durations, test_returns, avg_test_return = test_agent_on_env(
            env_name=env_name, agent=agent, num_episodes=100
        )

        log(
            f"[RESULT] Env: {env_name} | "
            f"Average Test Return: {avg_test_return:.2f} | "
            f"Expected (Target) Return: {target_reward:.2f}"
        )

        # 2.c. If agent passes the test
        if avg_test_return >= target_reward:
            log(
                f"[PASS] Env: {env_name} | Agent passed test "
                f"(AvgReturn={avg_test_return:.2f} >= Target={target_reward:.2f})"
            )

            # 2.c.1. Save models
            models_dir = os.path.join("trained_models", "PPO", env_name)
            os.makedirs(models_dir, exist_ok=True)
            actor_path = os.path.join(models_dir, "actor.pth")
            critic_path = os.path.join(models_dir, "critic.pth")
            agent.save_models(actor_path, critic_path)
            log(
                f"[SAVE] Saved models for {env_name} to {models_dir}"
            )

            # 2.c.2. Save test plots
            save_test_plots(env_name, test_durations, test_returns)

            # 2.c.3. Record 5 episodes as videos
            record_agent_videos(env_name, agent, num_episodes=5)
        else:
            log(
                f"[FAIL] Env: {env_name} | Agent did NOT pass test "
                f"(AvgReturn={avg_test_return:.2f} < Target={target_reward:.2f})"
            )
            log(
                f"[INFO] Stopping processing further environments."
            )
            break


if __name__ == "__main__":
    main()