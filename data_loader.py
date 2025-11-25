import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import logging

class NeuroSymbolicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, sym_module, knowledge_type):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sym_module = sym_module
        self.knowledge_type = knowledge_type
        self.needs_token_type = 'token_type_ids' in tokenizer.model_input_names

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        polarity = 0.0
        
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
            'raw_text': text # Needed for XAI
        }
        if self.needs_token_type:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
            
        return item

def get_dataloaders(config, tokenizer, sym_module, knowledge_type):
    logging.info(f"Loading dataset {config.DATASET_NAME}...")
    dataset = load_dataset(config.DATASET_NAME)
    
    train_ds = NeuroSymbolicDataset(
        dataset['train']['sentence'], dataset['train']['label'], 
        tokenizer, config.MAX_LEN, sym_module, knowledge_type
    )
    val_ds = NeuroSymbolicDataset(
        dataset['validation']['sentence'], dataset['validation']['label'], 
        tokenizer, config.MAX_LEN, sym_module, knowledge_type
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader