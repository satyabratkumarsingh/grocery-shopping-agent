
import ray
from firecrawl import FirecrawlApp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import json
import os
from time import sleep
import numpy as np
from cart_agent import CartItemAgent
from store_agent import StoreAgent
from state import get_state, CacheActor
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from gym.spaces import Discrete, Box
from grocery_env import GroceryMultiAgentEnv

def train_marl_rllib(cart_items, stores, user_preferences, num_episodes=100):
    ray.init(ignore_reinit_error=True)
    
    def env_creator(env_config):
        return GroceryMultiAgentEnv(cart_items, stores, user_preferences)
    register_env("grocery_multi_agent", env_creator)

    state_dim = len(cart_items) * len(stores) * 5 + len(stores)
    policies = {
        f"cart_item_{i}": (None, Box(low=0, high=np.inf, shape=(state_dim,), dtype=np.float32), Discrete(len(stores)), {})
        for i in range(len(cart_items))
    }
    policies.update({
        f"store_{i}": (None, Box(low=0, high=np.inf, shape=(state_dim,), dtype=np.float32), Discrete(len(cart_items)), {})
        for i in range(len(stores))
    })

    config = {
        "env": "grocery_multi_agent",
        "framework": "torch",
        "num_gpus": 1 if torch.cuda.is_available() else 0,
        "num_workers": 4,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
        "model": {
            "fcnet_hiddens": [128, 64],
        },
        "train_batch_size": 512,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "lr": 0.001,
        "clip_param": 0.2,
    }

    trainer = PPOTrainer(config=config)
    for episode in range(num_episodes):
        result = trainer.train()
        print(f"Episode {episode}: Mean Reward: {result['episode_reward_mean']}, Episode Length: {result['episode_len_mean']}")
        if episode % 10 == 0:
            print(f"Checkpoint saved at iteration {episode}")

    env = GroceryMultiAgentEnv(cart_items, stores, user_preferences)
    obs = env.reset()
    selected_stores = []
    for agent_id in [f"cart_item_{i}" for i in range(len(cart_items))]:
        action = trainer.compute_single_action(obs[agent_id], policy_id=agent_id)
        obs, rewards, dones, infos = env.step({agent_id: action})
        store_idx = action
        selected_stores.append(stores[store_idx]["store_name"])

    print("Selected stores for cart items:", selected_stores)
    return trainer, env
