
import ray
from firecrawl import FirecrawlApp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import json
import os
from time import sleep
import numpy as np
from grocery_env import GroceryEnv


@ray.remote
class CacheActor:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def put(self, key, value):
        self.cache[key] = value
        return True

def get_state(cart_items, store_agents, selected_stores, cache_actor):
    state = []
    cache_hits = 0
    cache_misses = 0
    
    try:
        for item in cart_items:
            base_product = item["base_product"]
            for store in store_agents:
                key = (base_product.replace(' ', '+'), str(store.store_name))
                print(f"Cache key: {key}")
                try:
                    store_data = ray.get(cache_actor.get.remote(key))
                    if store_data is None:
                        cache_misses += 1
                        env = GroceryEnv.remote(store.store_url, store.url_template)
                        store_data = ray.get(env.extract_data.remote([item]))
                        ray.get(cache_actor.put.remote(key, store_data))
                    else:
                        cache_hits += 1
                    if not isinstance(store_data, dict):
                        raise ValueError(f"Invalid store_data for key {key}: {store_data}")
                    data = store_data.get(base_product, {
                        "total_cost": 0.0,
                        "out_of_stock": True,
                        "delivery_time": 2,
                        "is_organic": False,
                        "quantity": 1,
                        "unit": item["unit"],
                        "cost_per_unit": float("inf")
                    })
                    state.extend([
                        float(data.get("total_cost", 0.0)),
                        1.0 if data.get("out_of_stock", True) else 0.0,
                        float(data.get("delivery_time", 2)),
                        1.0 if data.get("is_organic", False) else 0.0,
                        float(data.get("cost_per_unit", float("inf")))
                    ])
                except Exception as e:
                    print(f"Error processing item {base_product} for store {store.store_name}: {e}")
                    state.extend([0.0, 1.0, 2.0, 0.0, float("inf")])
        
        store_counts = [selected_stores.count(store.store_name) for store in store_agents]
        state.extend(store_counts)
        
        print(f"get_state: Cache hits={cache_hits}, misses={cache_misses}, cache size={ray.get(cache_actor.get.remote('size')) or 0}, state length={len(state)}")
        
        try:
            state_array = np.array(state, dtype=np.float32)
            expected_length = len(cart_items) * len(store_agents) * 5 + len(store_agents)
            if len(state_array) != expected_length:
                raise ValueError(f"Unexpected state length: {len(state_array)}, expected {expected_length}")
            return state_array
        except Exception as e:
            print(f"Error converting state to numpy array: {e}")
            raise RuntimeError(f"Invalid state vector: {state}")
    except Exception as e:
        print(f"Error in get_state: {e}")
        raise