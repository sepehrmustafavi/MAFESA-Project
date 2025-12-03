import torch
import logging
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class NeuroSymbolicDataset(Dataset):
    def __init__(self, dataset_name, data_split, tokenizer, max_len, sym_module, knowledge_type):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sym_module = sym_module
        self.knowledge_type = knowledge_type
        self.needs_token_type = 'token_type_ids' in tokenizer.model_input_names

        # Handle different column names
        if dataset_name == "sst2":
            self.texts = data_split['sentence']
            self.labels = data_split['label']
        elif dataset_name == "imdb":
            self.texts = data_split['text']
            self.labels = data_split['label']
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

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
    
    # Handling Dataset Loading
    if dataset_name == "sst2":
        dataset = load_dataset("stanfordnlp/sst2")
        train_split = dataset['train']
        val_split = dataset['validation']
        max_len = config.MAX_LEN_SST2
    elif dataset_name == "imdb":
        dataset = load_dataset("imdb")
        train_split = dataset['train']
        val_split = dataset['test'] 
        max_len = config.MAX_LEN_IMDB
    
    train_ds = NeuroSymbolicDataset(
        dataset_name, train_split, tokenizer, max_len, sym_module, knowledge_type
    )
    val_ds = NeuroSymbolicDataset(
        dataset_name, val_split, tokenizer, max_len, sym_module, knowledge_type
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, max_len