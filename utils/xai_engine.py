import torch
import logging
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

class XAIEngine:
    def __init__(self, device):
        self.device = device
        self.sbert = None
        #logging.info("SBERT loading skipped manually. Cosine Similarity will be 0.")
        #try:
            #self.sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        #except:
            #logging.warning("SBERT load failed. Cosine Similarity will be 0.")
            #self.sbert = None

    def get_prob(self, model, tokenizer, text, symbolic_polarity, max_len, target_class):
        """ Helper to get probability of a specific class for a text """
        if not text.strip(): return 0.5 # Neutral prior
        
        inputs = tokenizer.encode_plus(
            text, return_tensors='pt', max_length=max_len,
            padding='max_length', truncation=True
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attn_mask = inputs['attention_mask'].to(self.device)
        sym_tensor = torch.tensor([[symbolic_polarity]], dtype=torch.float).to(self.device)
        
        # Handle token_type_ids if model needs it (BERT)
        kwargs = {}
        if 'token_type_ids' in inputs and 'token_type_ids' in tokenizer.model_input_names:
            kwargs['token_type_ids'] = inputs['token_type_ids'].to(self.device)

        with torch.no_grad():
            outputs = model(input_ids, attn_mask, sym_tensor, **kwargs)
            probs = F.softmax(outputs, dim=1)
            return probs[0, target_class].item()

    def calculate_metrics(self, model, tokenizer, text, label, pred_class, keywords, sym_module, max_len):
        """ Calculates Sufficiency, Infidelity, and Cosine Similarity """
        model.eval()
        
        # 0. Base Probability
        # For XAI, we usually recalculate polarity for the specific text
        if hasattr(sym_module, 'get_text_polarity'):
             polarity, _ = sym_module.get_text_polarity(text)
        else:
             # For ConceptNet, simplistic re-calc
             polarity = 0.0 # Approximation for speed, or re-query if needed
             
        original_prob = self.get_prob(model, tokenizer, text, polarity, max_len, pred_class)

        # 1. Sufficiency: Keep ONLY keywords
        text_sufficiency = " ".join(keywords) if keywords else ""
        # Assumption: Polarity of keywords-only is similar to original or strictly based on them
        suff_prob = self.get_prob(model, tokenizer, text_sufficiency, polarity, max_len, pred_class)
        sufficiency_score = (original_prob - suff_prob) ** 2

        # 2. Infidelity: Remove keywords
        text_infidelity = text
        for k in keywords:
            text_infidelity = text_infidelity.replace(k, "")
        inf_prob = self.get_prob(model, tokenizer, text_infidelity, polarity, max_len, pred_class)
        infidelity_score = (original_prob - inf_prob) ** 2

        # 3. Cosine Similarity (Explanation Plausibility)
        cosine_sim = 0.0
        if self.sbert and keywords:
            emb_text = self.sbert.encode(text, convert_to_tensor=True)
            emb_expl = self.sbert.encode(" ".join(keywords), convert_to_tensor=True)
            cosine_sim = util.pytorch_cos_sim(emb_text, emb_expl).item()

        return {
            "sufficiency": sufficiency_score,
            "infidelity": infidelity_score,
            "cosine_similarity": cosine_sim,
            "keywords": keywords
        }