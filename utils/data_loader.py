import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import logging

class NeuroSymbolicDataset(Dataset):
    def __init__(self, dataset_name, data_split, tokenizer, max_len, sym_module, knowledge_type):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sym_module = sym_module
        self.knowledge_type = knowledge_type
        self.needs_token_type = 'token_type_ids' in tokenizer.model_input_names

        # Handle different column names
        # IMDB and SST-2 have different column names for text
        if "sst2" in dataset_name.lower():
            self.texts = data_split['sentence']
            self.labels = data_split['label']
        elif "imdb" in dataset_name.lower():
            self.texts = data_split['text']
            self.labels = data_split['label']
        else:
            raise ValueError(f"Unknown dataset format: {dataset_name}")

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        polarity = 0.0
        
        # Knowledge Extraction Logic
        if self.knowledge_type == 'senticnet':
            polarity, _ = self.sym_module.get_text_polarity(text)
        elif self.knowledge_type == 'conceptnet':
            keywords = self.sym_module.get_keywords_from_text(text)
            scores = []
            for k in keywords:
                s = self.sym_module.get_concept_score(k)
                if s != 0: scores.append(s)
            
            if scores:
                polarity = sum(scores) / len(scores)
            else:
                polarity = 0.0

        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True,
            return_token_type_ids=self.needs_token_type,
            return_attention_mask=True, return_tensors='pt',
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'symbolic_features': torch.tensor([polarity], dtype=torch.float),
            'raw_text': text 
        }
        if self.needs_token_type:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
            
        return item

def get_dataloaders(config, tokenizer, sym_module, knowledge_type, dataset_name):
    logging.info(f"Loading dataset: {dataset_name}...")
    
    # ----------------------------------------------------------------
    # DATASET LOADING LOGIC (Updated for stanfordnlp/imdb)
    # ----------------------------------------------------------------
    if dataset_name == "sst2":
        dataset = load_dataset("stanfordnlp/sst2")
        train_split = dataset['train']
        val_split = dataset['validation']
        max_len = config.MAX_LEN_SST2
        
    elif dataset_name == "imdb":
        # Using the specific ID as requested
        # 'stanfordnlp/imdb' typically maps to the canonical 'imdb' on HF
        try:
            dataset = load_dataset("stanfordnlp/imdb")
        except:
            # Fallback if the specific ID has issues, use canonical
            logging.warning("Could not load 'stanfordnlp/imdb', falling back to 'imdb'")
            dataset = load_dataset("imdb")
            
        train_split = dataset['train']
        # IMDB has 'test' split (25k) which we use for validation in this study
        val_split = dataset['test'] 
        max_len = config.MAX_LEN_IMDB
    
    else:
        raise ValueError("Invalid dataset name in scenario config.")
    
    # Create Datasets
    train_ds = NeuroSymbolicDataset(
        dataset_name, train_split, tokenizer, max_len, sym_module, knowledge_type
    )
    val_ds = NeuroSymbolicDataset(
        dataset_name, val_split, tokenizer, max_len, sym_module, knowledge_type
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, max_len