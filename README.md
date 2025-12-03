# MAFESA: Multi-Aspect Framework for Explainable Sentiment Analysis

This repository contains the implementation of the "Adaptive Neuro-Symbolic Framework" for Explainable Sentiment Analysis using TADA (Task-Agnostic Domain Adaptation).

## Project Structure
- `config.py`: Configuration settings and hyperparameters.
- `main.py`: Main entry point for training and evaluation.
- `knowledge_modules.py`: Integration with SenticNet and ConceptNet.
- `xai_engine.py`: Calculation of Sufficiency, Infidelity, and Cosine Similarity.

## Installation
1. Clone the repository.
2. Create the conda environment (Python 3.11)
   ```bash
   conda create --name MAFESA python=3.11
3. Install dependencies:
   ```bash
   pip install -r requirements.txt