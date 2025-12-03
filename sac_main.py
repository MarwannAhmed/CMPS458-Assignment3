import os
from collections import deque

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import wandb

from agents.sac_agent import SACAgent
from config import SAC_CONFIG, ENVIRONMENTS
from utils.defs import Transition
from utils.discrete_pendulum import DiscretePendulum


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
            name_prefix=f"SAC-{env_name}",
        )
    else:
        base_env = gym.make(env_name, render_mode="rgb_array")
        env = RecordVideo(
            base_env,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: True,
            name_prefix=f"SAC-{env_name}",
        )
    return env


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train_agent_on_env(
    env_name: str,
    max_episodes: int,
    target_reward: float,
    config: dict,
    n_steps: int = 5
):
    """
    Train an SAC agent on a given environment.
    Logs episodic statistics: duration, return, moving average, losses, etc.
    Returns: agent, training_stats (dict with logs)
    """
    log(f"========== Training on environment: {env_name} ==========")

    env = create_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    gamma = config["Gamma"]
    actor_lr = config["Actor LR"]
    critic1_lr = config["Critic 1 LR"]
    critic2_lr = config["Critic 2 LR"]
    entropy_coef = config["Entropy Coef"]
    hidden_dim = config["Hidden Dim"]

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        gamma=gamma,
        actor_lr=actor_lr,
        critic1_lr=critic1_lr,
        critic2_lr=critic2_lr,
        entropy_coef=entropy_coef,
    )

    run = wandb.init(project="cmps458_assignment3", name=f"SAC-{env_name}", config={**config, "env_id": env_name, "agent": "SAC"})

    episode_returns = []
    episode_durations = []
    moving_avg_returns = []
    actor_losses_per_episode = []
    critic1_losses_per_episode = []
    critic2_losses_per_episode = []
    recent_returns = deque(maxlen=100)

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        done = False
        t = 0

        raw_episode_return = 0.0  # unscaled return for logging/comparison
        agent.reset_memory()
        step_actor_losses = []
        step_critic1_losses = []
        step_critic2_losses = []

        while not done:
            action = agent.select_action(state)
            next_state, reward_env, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            t += 1
            raw_episode_return += reward_env

            reward = reward_env

            transition = Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=float(done),
            )
            agent.store_transition(transition)

            # N-step update if environment is not mountain car
            if (len(agent.memory) >= n_steps or done) and env_name != "MountainCar-v0":
                actor_loss, critic1_loss, critic2_loss = agent.update()
                agent.reset_memory()
                step_actor_losses.append(actor_loss)
                step_critic1_losses.append(critic1_loss)
                step_critic2_losses.append(critic2_loss)

            state = next_state
        
        # For MountainCar-v0, do update at end of episode
        if env_name == "MountainCar-v0" and len(agent.memory) > 0:
            actor_loss, critic1_loss, critic2_loss = agent.update()
            step_actor_losses.append(actor_loss)
            step_critic1_losses.append(critic1_loss)
            step_critic2_losses.append(critic2_loss)
            agent.reset_memory()

        episode_returns.append(raw_episode_return)
        episode_durations.append(t)
        recent_returns.append(raw_episode_return)

        avg_return_last_100 = np.mean(recent_returns)
        moving_avg_returns.append(avg_return_last_100)

        mean_actor_loss = np.mean(step_actor_losses) if step_actor_losses else 0.0
        mean_critic1_loss = np.mean(step_critic1_losses) if step_critic1_losses else 0.0
        mean_critic2_loss = np.mean(step_critic2_losses) if step_critic2_losses else 0.0
        actor_losses_per_episode.append(mean_actor_loss)
        critic1_losses_per_episode.append(mean_critic1_loss)
        critic2_losses_per_episode.append(mean_critic2_loss)

        log(
            f"[TRAIN] Env: {env_name} | Episode: {episode} | "
            f"Return: {raw_episode_return:.2f} | "
            f"Duration: {t} | "
            f"AvgReturn(100): {avg_return_last_100:.2f} | "
            f"ActorLoss: {mean_actor_loss:.6f} | Critic1Loss: {mean_critic1_loss:.6f} | Critic2Loss: {mean_critic2_loss:.6f}"
        )

        wandb.log({
            "episode": episode,
            "return": raw_episode_return,
            "avg100": avg_return_last_100,
            "actor_loss": mean_actor_loss,
            "critic1_loss": mean_critic1_loss,
            "critic2_loss": mean_critic2_loss,
            "steps": t
        })

    wandb.finish()
    env.close()

    training_stats = {
        "episode_returns": episode_returns,
        "episode_durations": episode_durations,
        "moving_avg_returns": moving_avg_returns,
        "actor_losses": actor_losses_per_episode,
        "critic1_losses": critic1_losses_per_episode,
        "critic2_losses": critic2_losses_per_episode,
    }

    return agent, training_stats


# ------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------
def test_agent_on_env(
    env_name: str,
    agent: SACAgent,
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
def save_test_plots(env_name: str, test_durations, test_returns, figures_root="figures/SAC"):
    """
    Generate and save:
    - Episode duration vs episode
    - Episode return vs episode
    in figures/SAC/<env_name>
    """
    env_fig_dir = os.path.join(figures_root, env_name)
    os.makedirs(env_fig_dir, exist_ok=True)

    episodes = np.arange(1, len(test_returns) + 1)

    # 1) Episode duration vs Episode
    plt.figure()
    plt.plot(episodes, test_durations)
    plt.xlabel("Episode")
    plt.ylabel("Episode Duration (steps)")
    plt.title(f"SAC - {env_name} - Episode Duration vs Episode")
    plt.grid(True)
    duration_path = os.path.join(env_fig_dir, "episode_duration_vs_episode.png")
    plt.savefig(duration_path)
    plt.close()

    # 2) Episode return vs Episode
    plt.figure()
    plt.plot(episodes, test_returns)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title(f"SAC - {env_name} - Episode Return vs Episode")
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
    agent: SACAgent,
    num_episodes: int = 5,
    videos_root: str = "videos/SAC",
):
    """
    Record num_episodes episodes of a trained agent and save videos into
    videos/SAC/<env_name>.
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
    
    # Iterate over all environments defined in ENVIRONMENTS / SAC_CONFIG
    for env_name, env_info in ENVIRONMENTS.items():
        if os.path.exists("trained_models/SAC/" + env_name + "/actor.pth") and os.path.exists("trained_models/SAC/" + env_name + "/critic1.pth") and os.path.exists("trained_models/SAC/" + env_name + "/critic2.pth"):
            log(
                f"Trained models for {env_name} already exist. Skipping training."
            )
            continue
        if env_name not in SAC_CONFIG:
            log(
                f"Environment {env_name} has no SAC_CONFIG entry. Skipping."
            )
            continue

        max_episodes = SAC_CONFIG[env_name]["Training Episodes"]
        target_reward = env_info["target_reward"]
        config = SAC_CONFIG[env_name]

        log(
            f"\n\n==================== ENVIRONMENT: {env_name} ====================\n"
        )
        log(
            f"Max Episodes: {max_episodes} | Target Reward: {target_reward} | "
            f"Gamma: {config['Gamma']} | Actor LR: {config['Actor LR']} | "
            f"Critic 1 LR: {config['Critic 1 LR']} | Critic 2 LR: {config['Critic 2 LR']} | Entropy Coef: {config['Entropy Coef']}"
        )

        # 2.a. Train the agent
        agent, training_stats = train_agent_on_env(
            env_name=env_name,
            max_episodes=max_episodes,
            target_reward=target_reward,
            config=config,
            n_steps=5,
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
            models_dir = os.path.join("trained_models", "SAC", env_name)
            os.makedirs(models_dir, exist_ok=True)
            actor_path = os.path.join(models_dir, "actor.pth")
            critic1_path = os.path.join(models_dir, "critic1.pth")
            critic2_path = os.path.join(models_dir, "critic2.pth")
            agent.save_models(actor_path, critic1_path, critic2_path)
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
