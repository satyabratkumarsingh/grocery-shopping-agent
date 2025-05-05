
import ray
from firecrawl import FirecrawlApp
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os
from time import sleep
import numpy as np


class CartItemAgent:
    def __init__(self, product, ppo_trainer, stores, store_agents, user_preferences):
        self.product = product["base_product"]
        self.quantity = product["quantity"]
        self.unit = product["unit"]
        self.ppo_trainer = ppo_trainer
        self.stores = stores
        self.store_agents = store_agents
        self.user_preferences = user_preferences
        self.selected_store = None

    def choose_action(self, state, current_selected_stores):
        try:
            store_counts = {store["store_name"]: current_selected_stores.count(store["store_name"]) for store in self.stores}
            preferred_store = max(store_counts, key=store_counts.get, default=self.stores[0]["store_name"])
            action_probs, _ = self.ppo_trainer.get_action_probabilities(state)
            action_probs_1d = action_probs.squeeze() if action_probs.ndim > 1 else action_probs
            if action_probs_1d.ndim != 1:
                raise ValueError(f"action_probs_1d must be 1D, got shape {action_probs_1d.shape}")
            action = np.random.choice(np.arange(len(self.stores)), p=action_probs_1d)
            if np.random.rand() < 0.7:
                action = next(i for i, store in enumerate(self.stores) if store["store_name"] == preferred_store)
            return action, action_probs_1d
        except Exception as e:
            print(f"Error in CartItemAgent choose_action: {e}")
            return 0, np.ones(len(self.stores)) / len(self.stores)

    def get_reward(self, state, action, selected_stores):
        try:
            store_agent = self.store_agents[action]
            store_data = ray.get(store_agent.env.extract_data.remote([{
                "base_product": self.product,
                "quantity": self.quantity,
                "unit": self.unit
            }]))
            data = store_data.get(self.product, {
                "total_cost": 0.0,
                "out_of_stock": True,
                "delivery_time": 2,
                "is_organic": False,
                "quantity": 1,
                "unit": self.unit,
                "cost_per_unit": float("inf")
            })
            price_score = -data["cost_per_unit"] * self.user_preferences["price_weight"]
            organic_score = (1.0 if data["is_organic"] else 0.0) * self.user_preferences["organic_weight"]
            delivery_score = -data["delivery_time"] * self.user_preferences["delivery_weight"]
            quantity_score = -abs(data["quantity"] - self.quantity)
            return price_score + organic_score + delivery_score + quantity_score
        except Exception as e:
            print(f"Error in CartItemAgent get_reward: {e}")
            return 0.0
