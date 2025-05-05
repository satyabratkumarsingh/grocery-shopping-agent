

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
from load_model import get_llm
from langchain_core.messages import HumanMessage
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
FIRECRAWL_KEY = os.getenv('FIRECRAWL_KEY')

os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY

@ray.remote
def extract(store_url, url_template, api_key, cart_items):
    store_data_product = {}
    app = FirecrawlApp(api_key=api_key)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    llm = get_llm()

    for item in cart_items:
        base_product = item["base_product"]
        search_terms = [
            base_product,
            f"{base_product} 1l", f"{base_product} 1.5 mobilizingl", f"{base_product} 2l",
            f"{base_product} 1kg", f"{base_product} 2kg"
        ] if base_product == "milk" else [base_product, f"{base_product} 1kg", f"{base_product} 2kg"]
        variants = []
        for term in search_terms:
            item_url = url_template.format(base_url=store_url.rstrip('/'), item=term.replace(' ', '+'))
            print(f"Calling URL {item_url}")
            for attempt in range(3):
                try:
                    crawl_result = app.scrape_url(url=item_url)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                    text_without_newlines = crawl_result.markdown.replace('\n', '')
                    chunks = text_splitter.create_documents([text_without_newlines])
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"Failed to scrape {item_url}: {e}")
                        variants.append({
                            "product": term,
                            "total_cost": 0.0,
                            "out_of_stock": True,
                            "delivery_time": 2,
                            "is_organic": False,
                            "quantity": "1",
                            "cost_per_unit": float("inf")
                        })
                    sleep(1)

            def build_prompt(text):
                return f"""
                You are an expert data extractor.
                Your task is to extract products and their prices from text. If you find any organic keyword in the text,
                set is_organic_value to "True", otherwise set it to "False".

                ONLY output in this strict JSON array format:
                [
                {{"product": "product_name", "price": "price_value", "quantity": "quantity_value", "is_organic": "is_organic_value"}}
                ]
                Ignore any filter, ad, or non-product data.
                Make sure to remove any qualifiers like "organic", "premium", "fresh", etc., and standardize the product_name.

                Here is the raw input:
                {text}
            """

            results = []
            for chunk in chunks:
                prompt = build_prompt(chunk.page_content)
                try:
                    # Use LangChain's invoke method for ChatOpenAI
                    response = llm.invoke(prompt)
                    print("@@@@@@ RESPONSE @@@@@@@@@@@")
                    print(response)
                    print('=======================')
                    extracted = json.loads(response)
                    print('@@@@@@ JSON LOADED @@@@@@@@@@')
                    print(extracted)
                    print('=======================')
                    results.extend(extracted)
                    break
                except (json.JSONDecodeError, Exception) as e:
                    continue
                    
            for result in results:
                if base_product.lower() in result["product"].lower():
                    try:
                        quantity_str = result["quantity"].lower()
                        quantity = float(quantity_str.replace("l", "").replace("kg", "").replace("lb", ""))
                        unit = "liter" if "l" in quantity_str else "kg" if "kg" in quantity_str else "lb"
                        total_cost = float(result["price"].replace("$", ""))
                        cost_per_unit = total_cost / quantity if quantity > 0 else float("inf")
                        variants.append({
                            "product": result["product"],
                            "total_cost": total_cost,
                            "out_of_stock": False,
                            "delivery_time": 2,
                            "is_organic": result["is_organic"] == "True",
                            "quantity": quantity,
                            "unit": unit,
                            "cost_per_unit": cost_per_unit
                        })
                    except (ValueError, KeyError) as e:
                        print(f"Error parsing item data for {term}: {e}")

        if variants:
            best_variant = min(variants, key=lambda x: x["cost_per_unit"] if not x["out_of_stock"] else float("inf"))
            store_data_product[base_product] = best_variant
        else:
            store_data_product[base_product] = {
                "product": base_product,
                "total_cost": 0.0,
                "out_of_stock": True,
                "delivery_time": 2,
                "is_organic": False,
                "quantity": 1,
                "unit": item["unit"],
                "cost_per_unit": float("inf")
            }
    print(store_data_product)
    return store_data_product
