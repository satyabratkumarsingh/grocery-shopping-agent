

import ray
from firecrawl import FirecrawlApp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import json
import os
from time import sleep
import numpy as np
from grocery_env import GroceryEnv

class StoreAgent:
    def __init__(self, store_name, store_url, user_preferences, cart_items, stores, ppo_trainer, url_template):
        self.store_name = store_name
        self.store_url = store_url
        self.user_preferences = user_preferences
        self.env = GroceryEnv.remote(store_url, url_template)
        self.cart_items = cart_items
        self.stores = stores
        self.ppo_trainer = ppo_trainer
        self.action_space = len(cart_items)

    def choose_action(self, state):
        if self.ppo_trainer is None:
            raise ValueError("Cannot choose action: ppo_trainer is None")
        try:
            action_probs, _ = self.ppo_trainer.get_action_probabilities(state)
            action_probs_1d = action_probs.squeeze() if action_probs.ndim > 1 else action_probs
            if action_probs_1d.ndim != 1:
                raise ValueError(f"action_probs_1d must be 1D, got shape {action_probs_1d.shape}")
            action = np.random.choice(np.arange(self.action_space), p=action_probs_1d)
            return action, action_probs_1d
        except Exception as e:
            print(f"Error in StoreAgent choose_action: {e}")
            return 0, np.ones(self.action_space) / self.action_space

    def get_reward(self, state, action):
        try:
            item = self.cart_items[action]
            store_data = ray.get(self.env.extract_data.remote([item]))
            data = store_data.get(item["base_product"], {
                "total_cost": 0.0,
                "out_of_stock": True,
                "delivery_time": 2,
                "is_organic": False,
                "quantity": 1,
                "unit": item["unit"],
                "cost_per_unit": float("inf")
            })
            price_score = -data["cost_per_unit"] * self.user_preferences["price_weight"]
            organic_score = (1.0 if data["is_organic"] else 0.0) * self.user_preferences["organic_weight"]
            delivery_score = -data["delivery_time"] * self.user_preferences["delivery_weight"]
            quantity_score = -abs(data["quantity"] - item["quantity"])
            return price_score + organic_score + delivery_score + quantity_score
        except Exception as e:
            print(f"Error in StoreAgent get_reward: {e}")
            return 0.0
        