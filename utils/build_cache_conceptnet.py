import time
import logging
import requests
import numpy as np
from tqdm import tqdm
import os, sys, time, json
from datasets import load_dataset
from .config import Config
from .knowledge_modules import ConceptNetModule

# Setup simple logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def build_cache():
    logging.info(f"Starting ConceptNet Cache Builder for dataset: {Config.DATASET_NAME}")
    
    # 1. Load Full Dataset (Train + Validation) to find ALL unique words
    dataset = load_dataset(Config.DATASET_NAME)
    all_sentences = dataset['train']['sentence'] + dataset['validation']['sentence']
    logging.info(f"Total sentences to process: {len(all_sentences)}")

    # 2. Initialize Module (loads existing cache if any)
    cn_module = ConceptNetModule(Config.CACHE_PATH)
    current_cache_size = len(cn_module.cache)
    logging.info(f"Current cache size: {current_cache_size}")

    # 3. Extract Unique Words
    unique_words = set()
    logging.info("Extracting unique words from dataset...")
    for text in tqdm(all_sentences, desc="Tokenizing"):
        words = cn_module.get_keywords_from_text(text)
        unique_words.update(words)
    
    logging.info(f"Total unique words found: {len(unique_words)}")

    # 4. Filter words that are NOT in cache
    words_to_fetch = [w for w in unique_words if w not in cn_module.cache]
    logging.info(f"New words to fetch from API: {len(words_to_fetch)}")

    if not words_to_fetch:
        logging.info("Cache is already up to date! No action needed.")
        return

    # 5. API Query Loop
    logging.info("Starting API queries (This may take a while)...")
    
    save_interval = 100
    counter = 0
    
    for word in tqdm(words_to_fetch, desc="Fetching from ConceptNet"):
        # We manually call the API logic here to ensure we save "not found" cases too
        # to avoid re-querying them later.
        try:
            # Query API
            response = requests.get(
                cn_module.api_url, 
                params={'node': f'/c/en/{word}', 'limit': 10}, 
                timeout=5
            )
            
            score = 0.0
            found = False
            
            if response.status_code == 200:
                data = response.json()
                edges = data.get('edges', [])
                if edges:
                    weights = [e['weight'] for e in edges]
                    raw_score = np.mean(weights)
                    # Normalize using tanh as defined in knowledge_modules.py
                    score = float(np.tanh(raw_score / 5.0))
                    found = True
            
            # Update Cache Dictionary directly
            cn_module.cache[word] = {
                "score": score,
                "found": found
            }
            
        except Exception as e:
            logging.warning(f"Error fetching '{word}': {e}")
            # Don't save failed requests to cache, so we retry next time
            pass
        
        # Rate Limiting (ConceptNet asks for 1 req/sec roughly, but we can burst a bit)
        time.sleep(0.1) 
        
        counter += 1
        if counter % save_interval == 0:
            save_json(cn_module.cache, Config.CACHE_PATH)

    # Final Save
    save_json(cn_module.cache, Config.CACHE_PATH)
    logging.info("\n*** Cache Build Complete Successfully ***")

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    build_cache()