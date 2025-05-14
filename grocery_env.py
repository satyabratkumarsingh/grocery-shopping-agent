
import ray
from ray import tune
from ray.rllib.env import MultiAgentEnv
import numpy as np
import gym
from gym.spaces import Discrete, Box
from firecrawl import FirecrawlApp
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from time import sleep
import os
from extract_data import extract
from dotenv import load_dotenv


load_dotenv()

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
FIRECRAWL_KEY = os.getenv('FIRECRAWL_KEY')

os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY


GLOBAL_CACHE = {}

class GroceryMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.cart_items = config["cart_items"]
        self.stores = config["stores"]
        self.user_preferences = config["user_preferences"]
        self.store_names = [store["store_name"] for store in self.stores]
        self.api_key = config.get("api_key", "mock_key")
        self.selected_stores = []
        self.proposals = np.zeros(len(self.stores))  # Store agent proposals
        self.current_step = 0

        # Action spaces
        self.action_space = {
            "cart": Discrete(len(self.stores)),  # Single cart agent
            **{f"store_{i}": Box(low=0, high=1000, shape=(1,)) for i in range(len(self.stores))}
        }

        # Observation spaces
        state_dim = len(self._get_state([]))
        self.observation_space = {
            agent: Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
            for agent in self.action_space.keys()
        }

        self.agents = list(self.action_space.keys())
        self._agent_ids = set(self.agents)

    def _get_state(self, selected_stores):
        state = []
        for item in self.cart_items:
            base_product = item["base_product"]
            for store in self.stores:
                store_name = store["store_name"]
                key = (base_product, store_name)
                print(key)
                if key not in GLOBAL_CACHE:
                    store_url = store["url"]
                    url_template = store["url_template"]
                    # Simulate ray.get for local execution
                    GLOBAL_CACHE[key] = extract.remote(store, [item])
                    print(GLOBAL_CACHE[key])
                store_data_ref = GLOBAL_CACHE[key]
                store_data = ray.get(store_data_ref)
                data = store_data.get(base_product, {
                    "price": 0.0,
                    "delivery_charge": 0.0,
                    "out_of_stock": True,
                    "delivery_time": 2,
                    "is_organic": False
                })
                state.extend([
                    data["price"],
                    1.0 if data["out_of_stock"] else 0.0,
                    data["delivery_time"],
                    1.0 if data["is_organic"] else 0.0,
                    data["delivery_charge"]
                ])
        store_counts = [selected_stores.count(name) for name in self.store_names]
        state.extend(store_counts)
        state.extend(self.proposals)
        return np.array(state)

    def _compute_reward(self, store_data, cart_items, selected_stores):
        reward = 0.0
        total_cost = 0.0
        delivery_cost = 0.0
        organic_matches = 0
        out_of_stock = False
        delivery_time = 2
        for item in cart_items:
            base_product = item["base_product"]
            quantity = item.get("quantity", 1)
            if base_product in store_data:
                data = store_data[base_product]
                total_cost += float(data["price"]) * quantity
                delivery_cost = max(delivery_cost, data.get("delivery_charge", 0.0))
                if data["out_of_stock"]:
                    out_of_stock = True
                if data["is_organic"] and self.user_preferences.get("prefer_organic", False):
                    organic_matches += 1
                delivery_time = max(delivery_time, data["delivery_time"])
        price_reward = -(total_cost + delivery_cost) * self.user_preferences.get("price_weight", 0.5)
        organic_reward = organic_matches * self.user_preferences.get("organic_weight", 0.3)
        delivery_penalty = -delivery_time * self.user_preferences.get("delivery_weight", 0.2)
        reward += price_reward + organic_reward + delivery_penalty
        if out_of_stock:
            reward -= 10.0
        if selected_stores:
            num_stores = len(set(selected_stores))
            if num_stores > 1:
                reward -= 50.0 * (num_stores - 1)  # Redundant with single cart agent
        return reward

    def reset(self):
        self.selected_stores = []
        self.proposals = np.zeros(len(self.stores))
        self.current_step = 0
        return {agent: self._get_state(self.selected_stores) for agent in self.agents}

    def step(self, action_dict):
        obs, rewards, dones, infos = {}, {}, {}, {}
        self.current_step += 1

        # Process actions
        for agent_id, action in action_dict.items():
            if agent_id.startswith("store_"):
                store_idx = int(agent_id.split("_")[1])
                store_name = self.stores[store_idx]["store_name"]
                proposed_cost = action[0]
                self.proposals[store_idx] = proposed_cost
                store_data = {}
                for item in self.cart_items:
                    key = (item["base_product"], store_name)
                    if key not in GLOBAL_CACHE:
                        store_url = self.stores[store_idx]["url"]
                        url_template = self.stores[store_idx]["url_template"]
                        GLOBAL_CACHE[key] = extract(store_url, url_template, self.api_key, [item])
                    store_data[item["base_product"]] = GLOBAL_CACHE[key][item["base_product"]]
                expected_cost = sum(float(data["price"]) * item.get("quantity", 1) 
                                  for item, data in zip(self.cart_items, store_data.values())) + \
                               max(data.get("delivery_charge", 0.0) for data in store_data.values())
                reward = self._compute_reward(store_data, self.cart_items, self.selected_stores)
                if abs(proposed_cost - expected_cost) > 1.0:
                    reward -= 10.0  # Penalize inaccurate proposals
                rewards[agent_id] = reward
            elif agent_id == "cart":
                store_idx = action
                store_name = self.stores[store_idx]["store_name"]
                self.selected_stores = [store_name]
                store_data = {}
                for item in self.cart_items:
                    key = (item["base_product"], store_name)
                    if key not in GLOBAL_CACHE:
                        store_url = self.stores[store_idx]["url"]
                        url_template = self.stores[store_idx]["url_template"]
                        GLOBAL_CACHE[key] = extract(store_url, url_template, self.api_key, [item])
                    store_data[item["base_product"]] = GLOBAL_CACHE[key][item["base_product"]]
                rewards[agent_id] = self._compute_reward(store_data, self.cart_items, self.selected_stores)

        state = self._get_state(self.selected_stores)
        for agent in self.agents:
            obs[agent] = state
            dones[agent] = self.current_step >= len(self.stores) + 1
            infos[agent] = {}

        dones["__all__"] = all(dones.values())
        return obs, rewards, dones, infos