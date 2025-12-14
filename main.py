import gc
import sys
import time
import json
import torch
import logging
import numpy as np
import os
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.config import Config
from utils.knowledge_modules import SymbolicModuleSenticNet, ConceptNetModule
from utils.model_architectures import NeuroSymbolicEncoder, NeuroSymbolicCausalLM, apply_tada
from utils.data_loader import get_dataloaders
from utils.xai_engine import XAIEngine

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler(Config.LOG_FILE), logging.StreamHandler(sys.stdout)]
)

# 2. Helper for Qualitative Samples (Updated with Natural Language Explanation)
def save_qualitative_samples(scenario_name, texts, labels, preds, keywords_list, filename="qualitative_analysis.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\nSCENARIO: {scenario_name}\n{'='*60}\n")
        n_samples = len(texts)
        if n_samples == 0: return

        # Select up to 5 random indices for display
        indices = np.random.choice(n_samples, min(5, n_samples), replace=False)
        
        for idx in indices:
            label_str = "Positive" if labels[idx] == 1 else "Negative"
            pred_str = "Positive" if preds[idx] == 1 else "Negative"
            keywords = keywords_list[idx]
            
            if keywords and len(keywords) > 0:
                kws_str = ", ".join([f"'{k}'" for k in keywords])
                explanation = (f"The model predicts '{pred_str}' primarily because "
                               f"it identified the following sentiment-bearing terms: {kws_str}.")
            else:
                explanation = (f"The model predicts '{pred_str}' based solely on neural analysis, "
                               "as no specific symbolic keywords were mapped.")
            
            f.write(f"Text: \"{texts[idx]}\"\n")
            f.write(f"Ground Truth: {label_str} | Model Prediction: {pred_str}\n")
            f.write(f"Explanation: {explanation}\n")
            f.write("-" * 50 + "\n")
    logging.info(f"Saved qualitative samples to {filename}")

# 3. Main Training Function
def run_training():
    results = []
    open("qualitative_analysis.txt", "w").close()
    xai_engine = XAIEngine(Config.DEVICE)
    
    for scenario in Config.SCENARIOS:
        sid, model_name, arch_type, strategy, k_source, ds_name = scenario
        scenario_id_str = f"Scenario {sid}: {model_name} | {strategy} | {k_source} | {ds_name}"
        
        logging.info(f"\n{'#'*60}\nSTARTING {scenario_id_str}\n{'#'*60}")

        # Init Knowledge & Tokenizer
        if k_source == 'senticnet': sym_module = SymbolicModuleSenticNet()
        else: sym_module = ConceptNetModule(Config.CACHE_PATH)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.padding_side = 'right'
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        except Exception as e: logging.error(f"Tokenizer error: {e}"); continue

        # Load Data
        train_loader, val_loader, current_max_len = get_dataloaders(Config, tokenizer, sym_module, k_source, ds_name)

        # Init Model
        try:
            if arch_type in ['bert', 'roberta']: model = NeuroSymbolicEncoder(model_name).to(Config.DEVICE)
            else: model = NeuroSymbolicCausalLM(model_name).to(Config.DEVICE)
            
            backbone = model.backbone
            if len(tokenizer) > backbone.get_input_embeddings().num_embeddings:
                backbone.resize_token_embeddings(len(tokenizer))
            apply_tada(model, arch_type, strategy)
        except Exception as e: logging.error(f"Model error: {e}"); continue

        # Optimizer
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE)
        total_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
        loss_fn = torch.nn.CrossEntropyLoss().to(Config.DEVICE)

        best_f1 = 0.0
        best_metrics = {}
        start_time = time.time()
        
        for epoch in range(Config.EPOCHS):
            # Training
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
                input_ids = batch['input_ids'].to(Config.DEVICE)
                mask = batch['attention_mask'].to(Config.DEVICE)
                sym = batch['symbolic_features'].to(Config.DEVICE)
                labels = batch['labels'].to(Config.DEVICE)
                
                kwargs = {}
                if 'token_type_ids' in batch: kwargs['token_type_ids'] = batch['token_type_ids'].to(Config.DEVICE)

                outputs = model(input_ids, mask, sym, **kwargs)
                loss = loss_fn(outputs, labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            preds, true_labels, val_texts = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(Config.DEVICE)
                    mask = batch['attention_mask'].to(Config.DEVICE)
                    sym = batch['symbolic_features'].to(Config.DEVICE)
                    kwargs = {}
                    if 'token_type_ids' in batch: kwargs['token_type_ids'] = batch['token_type_ids'].to(Config.DEVICE)
                    
                    outputs = model(input_ids, mask, sym, **kwargs)
                    preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    true_labels.extend(batch['labels'].numpy())
                    val_texts.extend(batch['raw_text'])

            acc = accuracy_score(true_labels, preds)
            p, r, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
            logging.info(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f} | F1={f1:.4f} | Acc={acc:.4f}")

            # Best Model Check & Full XAI Evaluation
            if f1 > best_f1:
                best_f1 = f1
                logging.info(f"--> New Best Model (F1: {best_f1:.4f}). Calculating FULL XAI (All Samples)...")
                
                xai_res = {"sufficiency": [], "infidelity": [], "cosine": []}
                qual_data = []

                # === FULL EVALUATION LOOP (No Sampling) ===
                # This will iterate over ALL validation samples (872 for SST2, ~25k for IMDB)
                for idx in tqdm(range(len(val_texts)), desc="Full XAI Calc", leave=False):
                    txt = val_texts[idx]
                    lbl = true_labels[idx]
                    prd = preds[idx]
                    
                    if k_source == 'senticnet': _, kws = sym_module.get_text_polarity(txt)
                    else: kws = sym_module.get_keywords_from_text(txt)
                        
                    metrics = xai_engine.calculate_metrics(model, tokenizer, txt, lbl, prd, kws, sym_module, current_max_len)
                    xai_res["sufficiency"].append(metrics["sufficiency"])
                    xai_res["infidelity"].append(metrics["infidelity"])
                    xai_res["cosine"].append(metrics["cosine_similarity"])
                    
                    # Keep first 20 samples to pick random qualitative ones later
                    if len(qual_data) < 20:
                        qual_data.append({"text": txt, "label": lbl, "pred": prd, "keywords": kws})

                best_metrics = {
                    "accuracy": acc, "precision": p, "recall": r, "f1": f1,
                    "avg_sufficiency": np.mean(xai_res["sufficiency"]),
                    "avg_infidelity": np.mean(xai_res["infidelity"]),
                    "avg_cosine": np.mean(xai_res["cosine"])
                }
                
                if qual_data:
                    # Pick 5 random samples from the collected batch
                    sel = np.random.choice(qual_data, min(5, len(qual_data)), replace=False)
                    save_qualitative_samples(scenario_id_str, [x['text'] for x in sel], [x['label'] for x in sel], [x['pred'] for x in sel], [x['keywords'] for x in sel])

        scenario_result = {
            "id": sid, "model": model_name, "strategy": strategy,
            "knowledge": k_source, "dataset": ds_name, "metrics": best_metrics,
            "training_time": time.time() - start_time
        }
        results.append(scenario_result)
        with open(Config.RESULTS_FILE, "w") as f: json.dump(results, f, indent=4)
        
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache(); gc.collect()
    
    logging.info("\nAll scenarios completed successfully.")

if __name__ == "__main__":
    run_training()
