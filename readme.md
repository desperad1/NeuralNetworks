# Neural Network Transformer Next-Word Prediction Reference

This repository showcases a carefully commented Transformer-based next-word prediction project suitable for interview discussions. The implementation is designed to be easy to follow and quick to train on modest hardware while demonstrating best practices for reproducible experiments.

## Project Structure

- `src/transformer_next_word.py` – Training script defining the dataset pipeline, Transformer model, and evaluation helper.

## Setup

1. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```
2. **Install dependencies**:
   ```bash
   pip install torch
   ```

## Usage

Train the model and run an example prediction:

```bash
python src/transformer_next_word.py
```

The script will report epoch losses and display the predicted next word for a sample prompt after training.

## Notes

- All code lines include inline comments to aid understanding during interviews.
- The small illustrative corpus keeps training time short while remaining relevant to neural network discussions.
