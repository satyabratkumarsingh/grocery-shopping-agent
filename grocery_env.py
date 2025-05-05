
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
        self.cart_items = config["cart_items"]  # List of dicts: [{"base_product": "apples", "unit": "kg"}, ...]
        self.stores = config["stores"]
        self.user_preferences = config["user_preferences"]
        self.store_names = [store["store_name"] for store in self.stores]
        self.api_key = config.get("api_key", FIRECRAWL_KEY)
        self.selected_stores = []
        self.current_step = 0

        # Define observation and action spaces
        state_dim = len(self._get_state([]))
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = {
            f"cart_{i}": Discrete(len(self.stores)) for i in range(len(self.cart_items))
        }
        self.action_space.update({
            f"store_{i}": Discrete(len(self.cart_items)) for i in range(len(self.stores))
        })

        self.agents = list(self.action_space.keys())
        self._agent_ids = set(self.agents)

    def _get_state(self, selected_stores):
        state = []
        for item in self.cart_items:
            base_product = item["base_product"]
            for store in self.stores:
                store_name = store["store_name"]
                key = (base_product, store_name)
                if key not in GLOBAL_CACHE:
                    store_url = store["url"]
                    url_template = store["url_template"]
                    # Fetch data remotely
                    store_data = ray.get(extract.remote(store_url, url_template, self.api_key, [item]))
                    GLOBAL_CACHE[key] = store_data
                store_data = GLOBAL_CACHE[key]
                data = store_data.get(base_product, {
                    "total_cost": 0.0,
                    "out_of_stock": True,
                    "delivery_time": 2,
                    "is_organic": False
                })
                state.extend([
                    data["total_cost"],
                    1.0 if data["out_of_stock"] else 0.0,
                    data["delivery_time"],
                    1.0 if data["is_organic"] else 0.0
                ])
        store_counts = [selected_stores.count(store_name) for store_name in self.store_names]
        state.extend(store_counts)
        return np.array(state)

    def _compute_reward(self, store_data, cart_items, selected_stores):
        reward = 0.0
        total_cost = 0.0
        organic_matches = 0
        out_of_stock = False
        delivery_time = 2
        for item in cart_items:
            base_product = item["base_product"]
            if base_product in store_data:
                data = store_data[base_product]
                total_cost += data["total_cost"]
                if data["out_of_stock"]:
                    out_of_stock = True
                if data["is_organic"] and self.user_preferences.get("prefer_organic", False):
                    organic_matches += 1
                delivery_time = max(delivery_time, data["delivery_time"])
        price_reward = -total_cost * self.user_preferences.get("price_weight", 0.5)
        organic_reward = organic_matches * self.user_preferences.get("organic_weight", 0.3)
        delivery_penalty = -delivery_time * self.user_preferences.get("delivery_weight", 0.2)
        reward += price_reward + organic_reward + delivery_penalty
        if out_of_stock:
            reward -= 10.0
        if selected_stores:
            num_stores = len(set(selected_stores))
            if num_stores > self.user_preferences.get("max_stores", 2):
                reward -= 5.0 * (num_stores - self.user_preferences["max_stores"])
        return reward

    def reset(self):
        self.selected_stores = []
        self.current_step = 0
        state = self._get_state(self.selected_stores)
        return {agent: state for agent in self.agents}

    def step(self, action_dict):
        obs, rewards, dones, infos = {}, {}, {}, {}
        self.current_step += 1

        for agent_id, action in action_dict.items():
            if agent_id.startswith("cart_"):
                store_idx = action
                store_name = self.stores[store_idx]["store_name"]
                self.selected_stores.append(store_name)
                item_idx = int(agent_id.split("_")[1])
                item = self.cart_items[item_idx]
                base_product = item["base_product"]
                key = (base_product, store_name)
                if key not in GLOBAL_CACHE:
                    store_url = self.stores[store_idx]["url"]
                    url_template = self.stores[store_idx]["url_template"]
                    GLOBAL_CACHE[key] = ray.get(extract.remote(store_url, url_template, self.api_key, [item]))
                store_data = GLOBAL_CACHE[key]
                rewards[agent_id] = self._compute_reward(store_data, [item], self.selected_stores)
            elif agent_id.startswith("store_"):
                item_idx = action
                store_idx = int(agent_id.split("_")[1])
                store_name = self.stores[store_idx]["store_name"]
                store_data = {}
                for item in self.cart_items:
                    base_product = item["base_product"]
                    key = (base_product, store_name)
                    if key not in GLOBAL_CACHE:
                        store_url = self.stores[store_idx]["url"]
                        url_template = self.stores[store_idx]["url_template"]
                        GLOBAL_CACHE[key] = ray.get(extract.remote(store_url, url_template, self.api_key, [item]))
                    store_data[base_product] = GLOBAL_CACHE[key][base_product]
                rewards[agent_id] = self._compute_reward(store_data, self.cart_items, self.selected_stores)

        state = self._get_state(self.selected_stores)
        for agent in self.agents:
            obs[agent] = state
            dones[agent] = self.current_step >= len(self.cart_items) + len(self.stores)
            infos[agent] = {}

        dones["__all__"] = all(dones.values())
        return obs, rewards, dones, infos
