import json
import os
import re
import numpy as np
import requests
import nltk
from nltk.corpus import stopwords
from senticnet.senticnet import SenticNet
import logging

# Download NLTK resources quietly
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class SymbolicModuleSenticNet:
    def __init__(self):
        try:
            self.sn = SenticNet()
        except Exception as e:
            logging.error(f"SenticNet load error: {e}")
            self.sn = None

    def get_text_polarity(self, text):
        if self.sn is None: return 0.0, []
        # Preprocessing
        text = re.sub(r'[^\w\s]', '', str(text))
        words = re.findall(r'\b\w+\b', text.lower())
        
        polarity_sum = 0.0
        found_keywords = []
        
        for word in words:
            try:
                # SenticNet returns a string like '0.123', convert to float
                pol = float(self.sn.polarity_value(word))
                polarity_sum += pol
                found_keywords.append(word)
            except:
                continue
        
        count = len(found_keywords)
        if count == 0:
            return 0.0, []
        
        # Average polarity clipped between -1 and 1
        avg_polarity = np.clip(polarity_sum / count, -1.0, 1.0)
        return avg_polarity, list(set(found_keywords))

class ConceptNetModule:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.stop_words = set(stopwords.words('english'))
        self.cache = self._load_cache()
        self.api_url = "http://api.conceptnet.io/query"

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logging.info(f"Loaded ConceptNet cache with {len(data)} words.")
                    return data
            except Exception as e:
                logging.warning(f"Cache load failed: {e}. Starting empty.")
                return {}
        return {}

    def get_keywords_from_text(self, text):
        text = re.sub(r'[^\w\s]', '', str(text))
        words = nltk.word_tokenize(text.lower())
        return list(set([w for w in words if w.isalpha() and w not in self.stop_words]))

    def get_concept_score(self, word):
        # 1. Check Cache
        if word in self.cache:
            item = self.cache[word]
            if item['found']:
                return item['score']
            return 0.0
        
        # 2. Fallback to API (Slow, but necessary if cache misses)
        try:
            response = requests.get(self.api_url, params={'node': f'/c/en/{word}', 'limit': 10}, timeout=2)
            if response.status_code == 200:
                edges = response.json().get('edges', [])
                if edges:
                    weights = [e['weight'] for e in edges]
                    score = np.mean(weights)
                    # Normalize roughly to [-1, 1] range (ConceptNet weights can be large)
                    score = np.tanh(score / 5.0) 
                    return score
        except:
            pass
        
        return 0.0