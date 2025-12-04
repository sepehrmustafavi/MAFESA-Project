import torch

class Config:
    # ---------------------------------------------------------
    # 1. General Settings
    # ---------------------------------------------------------
    PROJECT_NAME = "MAFESA_Experiment"
    DATASET_NAME = "stanfordnlp/sst2"
    RANDOM_STATE = 42
    
    # ---------------------------------------------------------
    # 2. GPU SETTINGS
    # ---------------------------------------------------------
    TARGET_GPU_ID = 7
    
    if torch.cuda.is_available():
        DEVICE = torch.device(f"cuda:{TARGET_GPU_ID}")
        try:
            _ = torch.tensor([1]).to(DEVICE)
        except:
            print(f"Warning: GPU {TARGET_GPU_ID} not found. Switching to default cuda:0")
            DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")
    
    # ---------------------------------------------------------
    # 3. Paths & Hyperparameters
    # ---------------------------------------------------------
    CACHE_PATH = "conceptnet_cache.json"
    LOG_FILE = "experiment_logs.txt"
    RESULTS_FILE = "final_results.json"

    BATCH_SIZE = 32      
    EPOCHS = 4           
    LEARNING_RATE = 2e-5 
    
    MAX_LEN_SST2 = 128
    MAX_LEN_IMDB = 256   
    
    # ---------------------------------------------------------
    # 4. Scenarios List
    # Format: (ID, Model, Arch, Strategy, Knowledge, DATASET_NAME)
    # ---------------------------------------------------------
    SCENARIOS = [
        # --- SST-2 Experiments (Main Study) ---
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
        # Comparing Static vs Flexible on a Harder Dataset
        (12, "Qwen/Qwen2-1.5B", "qwen2", "static", "senticnet", "imdb"),   # <--- NEW: Static Benchmark
        (13, "Qwen/Qwen2-1.5B", "qwen2", "flexible", "senticnet", "imdb")  # <--- NEW: Flexible (Expected Winner)
    ]