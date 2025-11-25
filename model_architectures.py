import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import logging

class NeuroSymbolicBase(nn.Module):
    def __init__(self, model_name, n_classes=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Handle hidden size for different architectures
        if hasattr(self.config, 'hidden_size'):
            self.hidden_dim = self.config.hidden_size
        elif hasattr(self.config, 'n_embd'): # GPT-2
            self.hidden_dim = self.config.n_embd
        else:
            self.hidden_dim = 768 # Default fallback
            
        # The '+1' is for the injected symbolic feature
        self.classifier = nn.Linear(self.hidden_dim + 1, n_classes)

    def forward_head(self, neural_features, symbolic_features):
        # Concatenate Neural (Batch, Hidden) + Symbolic (Batch, 1)
        combined = torch.cat([neural_features, symbolic_features], dim=1)
        return self.classifier(combined)

class NeuroSymbolicEncoder(NeuroSymbolicBase):
    """ For BERT, RoBERTa """
    def __init__(self, model_name, n_classes=2):
        super().__init__(model_name, n_classes)
        self.backbone = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, symbolic_features, token_type_ids=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Use CLS token
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.forward_head(cls_output, symbolic_features)

class NeuroSymbolicCausalLM(NeuroSymbolicBase):
    """ For GPT-2, Qwen, Llama """
    def __init__(self, model_name, n_classes=2):
        super().__init__(model_name, n_classes)
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model_type = self.config.model_type

    def forward(self, input_ids, attention_mask, symbolic_features, **kwargs):
        if self.model_type == 'gpt2':
            outputs = self.backbone(input_ids=input_ids)
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state
        # Extract last token representation using attention mask
        batch_size = input_ids.shape[0]
        # Calculate sequence lengths
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
        else:
            seq_lengths = torch.full((batch_size,), input_ids.shape[1] - 1).to(input_ids.device)
            
        # Ensure indices are within bounds
        seq_lengths = torch.clamp(seq_lengths, min=0, max=input_ids.shape[1]-1)
        
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        last_token_features = hidden_states[batch_indices, seq_lengths, :]
        
        return self.forward_head(last_token_features, symbolic_features)

def apply_tada(model, arch_type, strategy):
    """ Applies Freeze/Unfreeze logic based on strategy """
    logging.info(f"Configuring TADA: {strategy.upper()} for {arch_type}")
    
    # 1. Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Unfreeze Classifier (Always)
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 3. Unfreeze Embeddings (Always for TADA)
    backbone = model.backbone
    input_embeddings = backbone.get_input_embeddings()
    if input_embeddings:
        for param in input_embeddings.parameters():
            param.requires_grad = True
    else:
        logging.warning("Could not find input embeddings layer to unfreeze.")

    # 4. Unfreeze Last Layer (Only for Flexible)
    if strategy == 'flexible':
        last_layer = None
        try:
            if arch_type in ['bert', 'roberta']:
                last_layer = backbone.encoder.layer[-1]
            elif arch_type == 'gpt2':
                last_layer = backbone.h[-1]
            elif 'qwen' in arch_type:
                # Qwen structure usually backbone.layers or backbone.model.layers
                if hasattr(backbone, 'layers'):
                    last_layer = backbone.layers[-1]
                elif hasattr(backbone, 'model'):
                    last_layer = backbone.model.layers[-1]
            
            if last_layer:
                for param in last_layer.parameters():
                    param.requires_grad = True
                logging.info("-> Last Transformer Layer Unfrozen.")
            else:
                logging.warning(f"Could not identify last layer for {arch_type}")
        except Exception as e:
            logging.error(f"Error unfreezing last layer: {e}")

    # Log Stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable Params: {trainable:,} ({trainable/total:.2%})")