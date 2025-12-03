A2C_CONFIG = {
    "CartPole-v1": {
        "Gamma": 0.99,
        "Actor LR": 0.0004,
        "Critic LR": 0.001,
        "Hidden Dim": 128,
        "Training Episodes": 500
    },
    "Acrobot-v1": {
        "Gamma": 0.99,
        "Actor LR": 0.0005,
        "Critic LR": 0.001,
        "Hidden Dim": 128,
        "Training Episodes": 500
    },
    "MountainCar-v0": {
        "Gamma": 0.99,
        "Actor LR": 0.0004,
        "Critic LR": 0.001,
        "Hidden Dim": 128,
        "Training Episodes": 1000
    },
    "Pendulum-v1": {
        "Gamma": 0.95,
        "Actor LR": 0.0003,
        "Critic LR": 0.001,
        "Hidden Dim": 128,
        "Training Episodes": 500
    }
}

SAC_CONFIG = {
    "CartPole-v1": {
        "Gamma": 0.99,
        "Actor LR": 0.0004,
        "Critic 1 LR": 0.001,
        "Critic 2 LR": 0.001,
        "Entropy Coef": 0.01,
        "Hidden Dim": 128,
        "Training Episodes": 500
    },
    "Acrobot-v1": {
        "Gamma": 0.99,
        "Actor LR": 0.0005,
        "Critic 1 LR": 0.001,
        "Critic 2 LR": 0.001,
        "Entropy Coef": 0,
        "Hidden Dim": 128,
        "Training Episodes": 500
    },
    "MountainCar-v0": {
        "Gamma": 0.99,
        "Actor LR": 0.0004,
        "Critic 1 LR": 0.001,
        "Critic 2 LR": 0.001,
        "Entropy Coef": 0.1,
        "Hidden Dim": 128,
        "Training Episodes": 1000
    },
    "Pendulum-v1": {
        "Gamma": 0.97,
        "Actor LR": 0.00025,
        "Critic 1 LR": 0.0005,
        "Critic 2 LR": 0.0005,
        "Entropy Coef": 0.005,
        "Hidden Dim": 256,
        "Training Episodes": 1000
    }
}

PPO_CONFIG = {
    "CartPole-v1": {
        "Gamma": 0.99,
        "Actor LR": 0.0003,
        "Critic LR": 0.001,
        "Entropy Coef": 0.01,
        "Hidden Dim": 64,
        "Training Episodes": 500,
        "Rollout Length": 2048
    },
    "Acrobot-v1": {
        "Gamma": 0.99,
        "Actor LR": 0.0003,
        "Critic LR": 0.001,
        "Entropy Coef": 0.01,
        "Hidden Dim": 64,
        "Training Episodes": 500,
        "Rollout Length": 2048
    },
    "MountainCar-v0": {
        "Gamma": 0.99,
        "Actor LR": 0.0005,
        "Critic LR": 0.001,
        "Entropy Coef": 0.1,
        "Hidden Dim": 128,
        "Training Episodes": 2000,
        "Rollout Length": 4096
    },
    "Pendulum-v1": {
        "Gamma": 0.99,
        "Actor LR": 0.0003,
        "Critic LR": 0.001,
        "Entropy Coef": 0.01,
        "Hidden Dim": 128,
        "Training Episodes": 1000,
        "Rollout Length": 4096
    }
}

ENVIRONMENTS = {
    "CartPole-v1": {
        "target_reward": 475
    },
    "Acrobot-v1": {
        "target_reward": -100
    },
    "MountainCar-v0": {
        "target_reward": -110
    },
    "Pendulum-v1": {
        "target_reward": -200
    }
}