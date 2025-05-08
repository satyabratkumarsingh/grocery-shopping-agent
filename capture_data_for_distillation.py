import json
import os
from time import sleep
from firecrawl import FirecrawlApp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from load_model import get_llm
from langchain_core.messages import HumanMessage

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain.llms import HuggingFacePipeline
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
FIRECRAWL_KEY = os.getenv('FIRECRAWL_KEY')

os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY


llm = ChatOpenAI(
    api_key=OPEN_AI_KEY,
    model="gpt-3.5-turbo",
    temperature=0
)


cart_items = [
    {"base_product": "apples", "unit": "kg"},
    {"base_product": "milk", "unit": "liter"},
    {"base_product": "Eggs", "unit": "dozen"},
    {"base_product": "strawberry", "unit": "g"},
    {"base_product": "banana", "unit": "dozen"},
    {"base_product": "onion", "unit": "kg"},
    {"base_product": "potato", "unit": "kg"},
    {"base_product": "colgate", "unit": "ml"},
    {"base_product": "digestive bisuits", "unit": "g"},
    {"base_product": "bread", "unit": "g"},
    {"base_product": "carrots", "unit": "kg"},
    {"base_product": "rice", "unit": "kg"},
    {"base_product": "sausages", "unit": "g"},
    {"base_product": "butter", "unit": "g"},
    {"base_product": "cheese", "unit": "g"},
    {"base_product": "crisps", "unit": "g"},
    {"base_product": "salt", "unit": "g"},
    {"base_product": "suger", "unit": "g"},
    {"base_product": "yoghurt", "unit": "g"},
    {"base_product": "ham", "unit": "g"},
    {"base_product": "Coca-Cola Zero Sugar", "unit": "l"},
    {"base_product": "diet coke", "unit": "l"},
    {"base_product": "chicken", "unit": "kg"},
    {"base_product": "spinach", "unit": "g"},

]

stores = [
   
    {
        "name": "ASDA",
        "url": "https://groceries.asda.com/",
        "store_name": "ASDA",
        "url_template": "{base_url}/search/{item}"
    }
]


def extract(store,  firecrawl_api_key, cart_items, output_file='products.jsonl'):
    app = FirecrawlApp(api_key=firecrawl_api_key)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
   
    with open(output_file, 'w', encoding='utf-8') as f:
        pass

    for item in cart_items:
        base_product = item["base_product"]
        unit = item["unit"].lower()

        # Define quantity variations based on unit
        if unit in ["liter", "litre", "l"]:
            quantities = ["1l", "1.5l", "2l", "3l"]
        elif unit in ["kg", "kilogram", "kilograms"]:
            quantities = ["1kg", "2kg"]
        elif unit in ["g", "gram", "grams"]:
            quantities = ["100g", "200g", "225g", "300g", "400g", "500g", "600g", "500g", "1kg", "2kg", "30kg", "4kg", "5kg"]
        elif unit in ["ml", "milliliter", "milliliters"]:
            quantities = ["500ml", "1l", "75ml", "100ml", "200ml"]
        elif unit in ["lb", "pound", "pounds"]:
            quantities = ["1lb", "2lb"]
        elif unit in ["dozen", "count", "unit", "piece", "pieces"]:
            quantities = ["6", "12", "18", "24", "30"]
        else:
            quantities = []

        # Generate search terms
        search_terms = [base_product] + [f"{base_product} {q}" for q in quantities]

        store_url = store["url"]
        url_template = store["url_template"]

        for term in search_terms:
            item_url = url_template.format(base_url=store_url.rstrip('/'), item=term.replace(' ', '+'))
            print(f"Calling URL {item_url}")
            for attempt in range(3):
                try:
                    crawl_result = app.scrape_url(url=item_url)
                    text_without_newlines = crawl_result.markdown.replace('\n', '')
                    chunks = text_splitter.create_documents([text_without_newlines])
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"Failed to scrape {item_url}: {e}")
                    sleep(1)
            else:
                continue  # Skip to the next term if scraping failed

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

            for chunk in chunks:
                prompt = build_prompt(chunk.page_content)
                try:
                    print(prompt)
                    response = llm.invoke(prompt)
                    extracted = json.loads(response.content)
                    with open(output_file, 'a', encoding='utf-8') as f:
                        json_line = {
                            "input": chunk.page_content,
                            "output": extracted
                        }
                        f.write(json.dumps(json_line) + '\n')
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Error processing chunk: {e}")
                    continue

for store in stores:
    extract(store, FIRECRAWL_KEY, cart_items)