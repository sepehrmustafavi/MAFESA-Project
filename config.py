import torch

class Config:
    # General Settings
    PROJECT_NAME = "MAFESA_Experiment"
    RANDOM_STATE = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    CACHE_PATH = "conceptnet_cache.json"
    LOG_FILE = "experiment_logs.txt"
    RESULTS_FILE = "final_results.json"

    # Hyperparameters
    BATCH_SIZE = 32      # Reduced to 32 to contain IMDB longer sequences in GPU memory
    EPOCHS = 4           
    LEARNING_RATE = 2e-5 
    
    # Default Max Len (will be overridden for IMDB)
    MAX_LEN_SST2 = 128
    MAX_LEN_IMDB = 256   # IMDB reviews are longer
    
    # Scenarios List 
    # Format: (ID, Model, Arch, Strategy, Knowledge, DATASET_NAME)
    SCENARIOS = [
        # --- SST-2 Experiments (The Main Study) ---
        (1, "google-bert/bert-base-uncased", "bert", "static", "senticnet", "sst2"),
        (2, "google-bert/bert-base-uncased", "bert", "flexible", "senticnet", "sst2"),
        (3, "FacebookAI/roberta-base", "roberta", "static", "senticnet", "sst2"),
        (4, "FacebookAI/roberta-base", "roberta", "flexible", "senticnet", "sst2"),
        (5, "Qwen/Qwen2-0.5B", "qwen2", "static", "senticnet", "sst2"),
        (6, "Qwen/Qwen2-0.5B", "qwen2", "flexible", "senticnet", "sst2"),
        (7, "Qwen/Qwen2-1.5B", "qwen2", "static", "senticnet", "sst2"),
        (8, "Qwen/Qwen2-1.5B", "qwen2", "flexible", "senticnet", "sst2"),
        (9, "openai-community/gpt2-large", "gpt2", "static", "senticnet", "sst2"),
        (10, "openai-community/gpt2-large", "gpt2", "flexible", "senticnet", "sst2"),
        (11, "Qwen/Qwen2-1.5B", "qwen2", "flexible", "conceptnet", "sst2"),

        # --- IMDB Experiments (Robustness Check) ---
        # Testing the BEST model on a harder dataset
        (12, "Qwen/Qwen2-1.5B", "qwen2", "flexible", "senticnet", "imdb"),
        (13, "Qwen/Qwen2-1.5B", "qwen2", "flexible", "conceptnet", "imdb")
    ]