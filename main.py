import logging
import sys
import json
import torch
import time
import gc
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import Local Modules
from config import Config
from knowledge_modules import SymbolicModuleSenticNet, ConceptNetModule
from model_architectures import NeuroSymbolicEncoder, NeuroSymbolicCausalLM, apply_tada
from data_loader import get_dataloaders
from xai_engine import XAIEngine

# -------------------------------------------------------------------
# 1. Setup Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

# -------------------------------------------------------------------
# 2. Helper Function for Qualitative Analysis (Case Studies)
# -------------------------------------------------------------------
def save_qualitative_samples(scenario_name, texts, labels, preds, keywords_list, filename="qualitative_analysis.txt"):
    """
    Saves 5 random examples to a text file with Natural Language Explanations.
    Useful for Chapter 4 of the thesis (Case Studies).
    """
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\nSCENARIO: {scenario_name}\n{'='*60}\n")
        
        n_samples = len(texts)
        if n_samples == 0:
            return

        # Select up to 5 random indices
        indices = np.random.choice(n_samples, min(5, n_samples), replace=False)
        
        for idx in indices:
            label_str = "Positive" if labels[idx] == 1 else "Negative"
            pred_str = "Positive" if preds[idx] == 1 else "Negative"
            keywords = keywords_list[idx]
            
            # --- Generate Natural Language Explanation (Template-Based) ---
            if keywords and len(keywords) > 0:
                # Join keywords with commas
                kws_str = ", ".join([f"'{k}'" for k in keywords])
                explanation = (
                    f"The model predicts '{pred_str}' primarily because "
                    f"it identified the following sentiment-bearing terms: {kws_str}."
                )
            else:
                explanation = (
                    f"The model predicts '{pred_str}' based solely on neural analysis, "
                    "as no specific symbolic keywords were mapped from the knowledge base."
                )
            
            # Write to file
            f.write(f"Text: \"{texts[idx]}\"\n")
            f.write(f"Ground Truth: {label_str} | Model Prediction: {pred_str}\n")
            f.write(f"Explanation: {explanation}\n")
            f.write("-" * 50 + "\n")
            
    logging.info(f"Saved {min(5, n_samples)} qualitative samples to {filename}")

# -------------------------------------------------------------------
# 3. Main Training Function
# -------------------------------------------------------------------
def run_training():
    results = []
    
    # Clear qualitative file at the start
    open("qualitative_analysis.txt", "w").close()
    
    # Initialize XAI Engine once
    xai_engine = XAIEngine(Config.DEVICE)
    
    for scenario in Config.SCENARIOS:
        sid, model_name, arch_type, strategy, k_source = scenario
        scenario_id_str = f"Scenario {sid}: {model_name} | {strategy} | {k_source}"
        
        logging.info(f"\n{'#'*60}")
        logging.info(f"STARTING {scenario_id_str}")
        logging.info(f"{'#'*60}")

        # --- Step 1: Init Knowledge Module ---
        if k_source == 'senticnet':
            sym_module = SymbolicModuleSenticNet()
        else:
            sym_module = ConceptNetModule(Config.CACHE_PATH)

        # --- Step 2: Load Tokenizer ---
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.padding_side = 'right'
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logging.error(f"Tokenizer load failed for {model_name}: {e}")
            continue

        # --- Step 3: Load Data ---
        train_loader, val_loader = get_dataloaders(Config, tokenizer, sym_module, k_source)

        # --- Step 4: Initialize Model ---
        try:
            if arch_type in ['bert', 'roberta']:
                model = NeuroSymbolicEncoder(model_name).to(Config.DEVICE)
            else:
                model = NeuroSymbolicCausalLM(model_name).to(Config.DEVICE)
            
            # Resize embeddings if needed
            backbone = model.backbone
            input_embeddings = backbone.get_input_embeddings()
            if len(tokenizer) > input_embeddings.num_embeddings:
                logging.info(f"Resizing embeddings from {input_embeddings.num_embeddings} to {len(tokenizer)}")
                backbone.resize_token_embeddings(len(tokenizer))
                
            apply_tada(model, arch_type, strategy)
            
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            continue

        # --- Step 5: Optimizer & Scheduler ---
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE)
        total_steps = len(train_loader) * Config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * 0.1), 
            num_training_steps=total_steps
        )
        loss_fn = torch.nn.CrossEntropyLoss().to(Config.DEVICE)

        # --- Step 6: Training Loop ---
        best_f1 = 0.0
        best_metrics = {}
        
        start_time = time.time()
        
        for epoch in range(Config.EPOCHS):
            # A. Training
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
                input_ids = batch['input_ids'].to(Config.DEVICE)
                mask = batch['attention_mask'].to(Config.DEVICE)
                sym = batch['symbolic_features'].to(Config.DEVICE)
                labels = batch['labels'].to(Config.DEVICE)
                
                kwargs = {}
                if 'token_type_ids' in batch:
                    kwargs['token_type_ids'] = batch['token_type_ids'].to(Config.DEVICE)

                outputs = model(input_ids, mask, sym, **kwargs)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # B. Validation
            model.eval()
            preds, true_labels = [], []
            val_texts = [] 
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(Config.DEVICE)
                    mask = batch['attention_mask'].to(Config.DEVICE)
                    sym = batch['symbolic_features'].to(Config.DEVICE)
                    
                    kwargs = {}
                    if 'token_type_ids' in batch:
                        kwargs['token_type_ids'] = batch['token_type_ids'].to(Config.DEVICE)
                        
                    outputs = model(input_ids, mask, sym, **kwargs)
                    batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    
                    preds.extend(batch_preds)
                    true_labels.extend(batch['labels'].numpy())
                    val_texts.extend(batch['raw_text'])

            # Calculate Standard Metrics
            acc = accuracy_score(true_labels, preds)
            p, r, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
            
            logging.info(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f} | F1={f1:.4f} | Acc={acc:.4f}")

            # C. Check Best Model & Run Full XAI Evaluation
            if f1 > best_f1:
                best_f1 = f1
                logging.info(f"--> New Best Model (F1: {best_f1:.4f}). Calculating FULL XAI metrics (This may take time)...")
                
                xai_res = {"sufficiency": [], "infidelity": [], "cosine": []}
                
                # Arrays for qualitative sampling
                qual_keywords = []
                qual_texts_subset = []
                qual_labels = []
                qual_preds = []

                # === FULL DATASET EVALUATION ===
                # We iterate over ALL validation samples
                # Using tqdm to show progress because this will be slow
                
                for idx in tqdm(range(len(val_texts)), desc="XAI Calc", leave=False):
                    txt = val_texts[idx]
                    lbl = true_labels[idx]
                    prd = preds[idx]
                    
                    # Extract keywords
                    if k_source == 'senticnet':
                        _, kws = sym_module.get_text_polarity(txt)
                    else:
                        kws = sym_module.get_keywords_from_text(txt)
                        
                    # Calculate Metrics
                    metrics = xai_engine.calculate_metrics(
                        model, tokenizer, txt, lbl, prd, kws, sym_module, Config.MAX_LEN
                    )
                    xai_res["sufficiency"].append(metrics["sufficiency"])
                    xai_res["infidelity"].append(metrics["infidelity"])
                    xai_res["cosine"].append(metrics["cosine_similarity"])
                    
                    # Collect first 5 samples for qualitative analysis
                    if len(qual_texts_subset) < 5:
                        qual_keywords.append(kws)
                        qual_texts_subset.append(txt)
                        qual_labels.append(lbl)
                        qual_preds.append(prd)

                # Store Best Metrics (Averaged over FULL dataset)
                best_metrics = {
                    "accuracy": acc, "precision": p, "recall": r, "f1": f1,
                    "avg_sufficiency": np.mean(xai_res["sufficiency"]),
                    "avg_infidelity": np.mean(xai_res["infidelity"]),
                    "avg_cosine": np.mean(xai_res["cosine"])
                }
                
                logging.info(f"--> Full XAI Scores: Suff={best_metrics['avg_sufficiency']:.4f}, Infid={best_metrics['avg_infidelity']:.4f}")

                # Save Qualitative Samples
                save_qualitative_samples(
                    scenario_id_str, qual_texts_subset, qual_labels, qual_preds, qual_keywords
                )

        # --- End of Scenario ---
        scenario_result = {
            "id": sid, "model": model_name, "strategy": strategy,
            "knowledge": k_source, "metrics": best_metrics,
            "training_time": time.time() - start_time
        }
        results.append(scenario_result)
        
        with open(Config.RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=4)
        
        logging.info(f"Finished {scenario_id_str}. Best F1: {best_f1:.4f}")

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    logging.info("\nAll scenarios completed successfully.")

if __name__ == "__main__":
    run_training()