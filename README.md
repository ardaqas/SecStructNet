# 🔬 SecStructNet — CNN-Based Protein Secondary Structure Predictor


## 🔖 Version 1.0

`SecStructNet` is a deep learning model that predicts **protein secondary structure** — `H` (Helix), `E` (Sheet), `C` (Coil) — using a 1D CNN architecture trained on amino acid sequences.

This repo includes:
- Full training pipeline (PyTorch)
- Padding-aware per-residue classification
- Evaluation with Q3 score + confusion matrix
- Inference mode with raw amino acid input
- Loss & accuracy visualizations

---
SecStructNet/
- model.py           # CNN architecture
- dataset.py         # Encoding & preprocessing
- train.py           # Model training
- inference.py       # Predict structure from raw sequence
- evaluate.py        # Eval metrics & confusion matrix
- config.py          # Constants & hyperparams
- requirements.txt   # All dependencies
