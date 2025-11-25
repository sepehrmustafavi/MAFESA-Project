import torch

class Config:
    # General Settings
    PROJECT_NAME = "MAFESA_Experiment"
    DATASET_NAME = "stanfordnlp/sst2"
    RANDOM_STATE = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    CACHE_PATH = "conceptnet_cache.json"
    LOG_FILE = "experiment_logs.txt"
    RESULTS_FILE = "final_results.json"

    # Hyperparameters
    BATCH_SIZE = 64      # Lower this if GPU OOM occurs (e.g., to 32 or 16)
    EPOCHS = 4           # Epochs per scenario
    LEARNING_RATE = 2e-5 # Standard for Fine-tuning
    MAX_LEN = 128
    
    # Scenarios List (ID, Model Name, Arch Type, TADA Strategy, Knowledge Source)
    SCENARIOS = [
        (1, "google-bert/bert-base-uncased", "bert", "static", "senticnet"),
        (2, "google-bert/bert-base-uncased", "bert", "flexible", "senticnet"),
        (3, "FacebookAI/roberta-base", "roberta", "static", "senticnet"),
        (4, "FacebookAI/roberta-base", "roberta", "flexible", "senticnet"),
        (5, "Qwen/Qwen2-0.5B", "qwen2", "static", "senticnet"),
        (6, "Qwen/Qwen2-0.5B", "qwen2", "flexible", "senticnet"),
        (7, "Qwen/Qwen2-1.5B", "qwen2", "static", "senticnet"),
        (8, "Qwen/Qwen2-1.5B", "qwen2", "flexible", "senticnet"),
        (9, "openai-community/gpt2-large", "gpt2", "static", "senticnet"),
        (10, "openai-community/gpt2-large", "gpt2", "flexible", "senticnet"),
        (11, "Qwen/Qwen2-1.5B", "qwen2", "flexible", "conceptnet")
    ]