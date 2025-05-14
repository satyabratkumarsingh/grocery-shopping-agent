
import ray
from ray import tune
from ray.rllib.algorithms import Algorithm
from ray.rllib.env import MultiAgentEnv
import numpy as np
import gym
from gym.spaces import Discrete, Box
from firecrawl import FirecrawlApp
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from time import sleep
from grocery_env import GroceryMultiAgentEnv
import os
from dotenv import load_dotenv


load_dotenv()

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
FIRECRAWL_KEY = os.getenv('FIRECRAWL_KEY')

os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY

# Mock data for testing
cart_items = [
    {"base_product": "apples", "quantity": "1kg"},
    {"base_product": "milk", "quantity": "1l"}
]
stores = [
    {
        "name": "Tesco",
        "url": "https://www.tesco.com",
        "store_name": "Tesco",
        "url_template": "{base_url}/groceries/en-GB/search?query={item}"
    },
    {
        "name": "Ocado",
        "url": "https://www.ocado.com",
        "store_name": "Ocado",
        "url_template": "{base_url}/search?entry={item}"
    }
]
user_preferences = {
    "price_weight": 0.5,
    "organic_weight": 0.3,
    "delivery_weight": 0.2,
    "prefer_organic": True,
    "max_stores": 2
}

config = {
    "env": GroceryMultiAgentEnv,
    "env_config": {
        "cart_items": cart_items,
        "stores": stores,
        "user_preferences": user_preferences,
        "api_key": FIRECRAWL_KEY
    },
    "framework": "torch",
    "num_workers": 2,
    "num_gpus": 0,
    "model": {
        "fcnet_hiddens": [128, 64],
        "fcnet_activation": "relu"
    },
    "multiagent": {
        "policies": {
            **{f"cart_{i}": (
                None,
                Box(low=-np.inf, high=np.inf, shape=(len(GroceryMultiAgentEnv({"cart_items": cart_items, "stores": stores, "user_preferences": user_preferences})._get_state([])),), dtype=np.float32),
                Discrete(len(stores)),
                {}
            ) for i in range(len(cart_items))},
            **{f"store_{i}": (
                None,
                Box(low=-np.inf, high=np.inf, shape=(len(GroceryMultiAgentEnv({"cart_items": cart_items, "stores": stores, "user_preferences": user_preferences})._get_state([])),), dtype=np.float32),
                Discrete(len(cart_items)),
                {}
            ) for i in range(len(stores))}
        },
        "policy_mapping_fn": lambda agent_id: agent_id
    },
    "rollout_fragment_length": 200,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 128,
    "num_sgd_iter": 10,
    "lr": 0.001,
    "algorithm": "PPO"  # Specify PPO algorithm
}

# Train with RLlib's Algorithm
trainer = Algorithm(config=config)

num_episodes = 20
for i in range(num_episodes):
    result = trainer.train()
    print(f"Iteration {i}, Mean Reward: {result['episode_reward_mean']}")
    if i % 10 == 0:
        print(f"Iteration {i}, Training Info: {result}")

# Inference
env = GroceryMultiAgentEnv({
    "cart_items": cart_items,
    "stores": stores,
    "user_preferences": user_preferences,
    "api_key": FIRECRAWL_KEY
})
obs = env.reset()
selected_stores = []
done = False
while not done:
    action_dict = {}
    for agent_id, agent_obs in obs.items():
        policy = trainer.compute_single_action(agent_obs, policy_id=agent_id)
        action_dict[agent_id] = policy
    obs, rewards, dones, infos = env.step(action_dict)
    for agent_id, action in action_dict.items():
        if agent_id.startswith("cart_"):
            store_idx = action
            selected_stores.append(stores[store_idx]["store_name"])
    done = dones["__all__"]

print("Selected stores for cart items:", selected_stores)

# Shutdown Ray
ray.shutdown()
