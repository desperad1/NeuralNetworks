"""Entry point for training and evaluating a Transformer next-word prediction model."""  # Module docstring describing the purpose of this script.
import math  # Import math for square root operations in learning rate scheduling.
import random  # Import random for deterministic shuffling of data.
from typing import Dict, List, Tuple  # Import typing helpers for type annotations.
import torch  # Import PyTorch for tensor computations and model training.
from torch import nn  # Import neural network modules from PyTorch.
from torch.optim import AdamW  # Import AdamW optimizer for stable training.
from torch.utils.data import DataLoader, Dataset  # Import dataset utilities for batching.
def set_seed(seed: int) -> None:  # Define a utility to set deterministic seeds.
    """Seed all random number generators for reproducibility."""  # Provide a concise docstring for this helper.
    random.seed(seed)  # Seed the Python random module for reproducible shuffles.
    torch.manual_seed(seed)  # Seed PyTorch for deterministic behavior on CPU.
class Vocabulary:  # Define a simple vocabulary to map words to indices.
    """A minimal vocabulary supporting token-to-index conversions."""  # Describe the purpose of the class.
    def __init__(self) -> None:  # Initialize the vocabulary data structures.
        self.token_to_index: Dict[str, int] = {}  # Dictionary mapping tokens to integer indices.
        self.index_to_token: List[str] = []  # List storing tokens by index position.
    def add_token(self, token: str) -> int:  # Add a token to the vocabulary if unseen.
        if token not in self.token_to_index:  # Check whether the token already exists.
            index = len(self.index_to_token)  # Compute the next available index.
            self.token_to_index[token] = index  # Store the mapping from token to index.
            self.index_to_token.append(token)  # Append the token for reverse lookup.
        return self.token_to_index[token]  # Return the index for the provided token.
    def token_to_id(self, token: str) -> int:  # Retrieve the index of a token.
        return self.token_to_index[token]  # Return the token index directly.
    def id_to_token(self, index: int) -> str:  # Retrieve a token by index.
        return self.index_to_token[index]  # Return the token from the indexed list.
    def __len__(self) -> int:  # Provide the vocabulary size when len() is called.
        return len(self.index_to_token)  # Return the total number of tracked tokens.
class NextWordDataset(Dataset):  # Create a dataset for next-word prediction.
    """Dataset producing input and target sequences for language modeling."""  # Explain the dataset role.
    def __init__(self, sequences: List[List[int]], context_length: int) -> None:  # Initialize with tokenized sequences.
        self.samples: List[Tuple[List[int], int]] = []  # Prepare storage for context-target pairs.
        for sequence in sequences:  # Iterate over each tokenized sentence.
            if len(sequence) <= context_length:  # Skip sequences shorter than the context length.
                continue  # Continue to the next sequence because no target can be formed.
            for index in range(context_length, len(sequence)):  # Slide a window across each sequence.
                context = sequence[index - context_length : index]  # Extract the context tokens before the target.
                target = sequence[index]  # Identify the next token as the prediction target.
                self.samples.append((context, target))  # Store the context-target pair in the dataset.
    def __len__(self) -> int:  # Return dataset length for batching.
        return len(self.samples)  # Provide the number of context-target pairs.
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # Retrieve a sample at the specified index.
        context, target = self.samples[index]  # Unpack the stored context and target.
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)  # Convert to tensors for PyTorch.
class PositionalEncoding(nn.Module):  # Implement sinusoidal positional encoding for sequence order information.
    """Sinusoidal positional encoding compatible with Transformer inputs."""  # Describe the encoding purpose.
    def __init__(self, embedding_dim: int, max_length: int = 5000) -> None:  # Initialize with embedding dimension and maximum length.
        super().__init__()  # Initialize the nn.Module superclass.
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)  # Create a column vector of positions.
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))  # Compute frequency divisors for sine and cosine.
        pe = torch.zeros(max_length, embedding_dim)  # Allocate the positional encoding tensor.
        pe[:, 0::2] = torch.sin(position * div_term)  # Populate even dimensions with sine components.
        pe[:, 1::2] = torch.cos(position * div_term)  # Populate odd dimensions with cosine components.
        self.register_buffer("pe", pe.unsqueeze(0))  # Register the encoding as a buffer to avoid optimization.
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Apply positional encoding to input embeddings.
        return x + self.pe[:, : x.size(1)]  # Add positional encoding to match the input sequence length.
class TransformerLanguageModel(nn.Module):  # Define the Transformer model for next-word prediction.
    """Transformer-based language model predicting next tokens."""  # Describe the model capability.
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        context_length: int,
    ) -> None:  # Initialize the Transformer with configurable hyperparameters.
        super().__init__()  # Initialize the nn.Module superclass.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Create an embedding layer converting tokens to vectors.
        self.positional_encoding = PositionalEncoding(embedding_dim, max_length=context_length)  # Instantiate positional encoding for the context length.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )  # Configure a single Transformer encoder layer with the provided settings.
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # Stack multiple encoder layers.
        self.output_layer = nn.Linear(embedding_dim, vocab_size)  # Map encoder outputs to vocabulary logits.
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # Define the forward pass producing logits for each position.
        embeddings = self.embedding(input_ids)  # Convert token ids to embeddings.
        embeddings = self.positional_encoding(embeddings)  # Inject positional information into embeddings.
        encoded = self.transformer_encoder(embeddings)  # Process the sequence through the Transformer encoder.
        logits = self.output_layer(encoded[:, -1, :])  # Compute logits for the final position to predict the next token.
        return logits  # Return the vocabulary logits for the next token.
def build_vocabulary(corpus: List[str]) -> Tuple[Vocabulary, List[List[int]]]:  # Create a vocabulary and tokenize the corpus.
    vocab = Vocabulary()  # Initialize an empty vocabulary.
    tokenized_sequences: List[List[int]] = []  # Prepare storage for tokenized sentences.
    for sentence in corpus:  # Iterate over each sentence in the corpus.
        tokens = sentence.lower().split()  # Tokenize by splitting on whitespace and lowercasing.
        indices = [vocab.add_token(token) for token in tokens]  # Convert tokens to indices, adding new ones as needed.
        tokenized_sequences.append(indices)  # Store the tokenized sentence.
    return vocab, tokenized_sequences  # Return the constructed vocabulary and tokenized data.
def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:  # Prepare batches of contexts and targets.
    contexts = torch.stack([item[0] for item in batch])  # Stack context tensors along the batch dimension.
    targets = torch.stack([item[1] for item in batch])  # Stack target tensors into a batch vector.
    return contexts, targets  # Return batched contexts and targets.
def generate_next_word(model: TransformerLanguageModel, prompt: str, vocab: Vocabulary, context_length: int, device: torch.device) -> str:  # Generate the next word from a prompt using the trained model.
    tokens = prompt.lower().split()  # Tokenize the prompt into lowercase tokens.
    context_tokens = tokens[-context_length:]  # Extract the last tokens up to the context length.
    context_indices = [vocab.token_to_id(token) for token in context_tokens if token in vocab.token_to_index]  # Convert known tokens to indices.
    if len(context_indices) < context_length:  # Ensure the context matches the expected length.
        padding = [0] * (context_length - len(context_indices))  # Create padding indices for missing context.
        context_indices = padding + context_indices  # Prepend padding to align the context length.
    input_tensor = torch.tensor(context_indices[-context_length:], dtype=torch.long).unsqueeze(0).to(device)  # Build a batch of size one for inference.
    model.eval()  # Switch the model to evaluation mode for deterministic behavior.
    with torch.no_grad():  # Disable gradient computation during inference.
        logits = model(input_tensor)  # Obtain logits for the next token from the model.
        probabilities = torch.softmax(logits, dim=-1)  # Convert logits to probabilities using softmax.
        predicted_index = torch.argmax(probabilities, dim=-1).item()  # Select the token with the highest probability.
    return vocab.id_to_token(predicted_index)  # Convert the predicted index back to a token string.
def train_model() -> None:  # Main training routine orchestrating data preparation, training, and evaluation.
    set_seed(42)  # Ensure reproducibility across runs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select GPU if available, otherwise CPU.
    corpus = [
        "deep learning models transform data into insights",
        "transformers excel at capturing long range dependencies",
        "neural networks can approximate complex functions",
        "attention mechanisms help models focus on relevant information",
        "language models predict the next word in a sentence",
        "training data quality influences model performance",
        "gradient descent optimizes neural network parameters",
        "model documentation improves knowledge transfer across teams",
        "pytorch provides flexible tools for building models",
        "researchers continue improving transformer architectures",
    ]  # Define a small illustrative corpus of domain-relevant sentences.
    context_length = 4  # Specify the number of tokens used as context for prediction.
    vocab, tokenized_sequences = build_vocabulary(corpus)  # Build vocabulary and tokenize the corpus.
    dataset = NextWordDataset(tokenized_sequences, context_length=context_length)  # Create a dataset for language modeling.
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_batch)  # Prepare a data loader for batching and shuffling.
    model = TransformerLanguageModel(
        vocab_size=len(vocab),
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        feedforward_dim=128,
        dropout=0.1,
        context_length=context_length,
    ).to(device)  # Instantiate the Transformer model with reasonable hyperparameters.
    criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss for classification over the vocabulary.
    optimizer = AdamW(model.parameters(), lr=2e-3)  # Configure the AdamW optimizer for stable convergence.
    num_epochs = 30  # Define the number of training epochs for the small dataset.
    for epoch in range(num_epochs):  # Iterate through each training epoch.
        model.train()  # Set the model to training mode to enable dropout and gradients.
        epoch_loss = 0.0  # Initialize the running loss for the epoch.
        for contexts, targets in dataloader:  # Iterate over mini-batches from the data loader.
            contexts = contexts.to(device)  # Move context tensors to the computation device.
            targets = targets.to(device)  # Move target tensors to the computation device.
            optimizer.zero_grad()  # Reset gradients from the previous iteration.
            logits = model(contexts)  # Forward pass to compute logits for each batch.
            loss = criterion(logits, targets)  # Calculate the loss between predictions and targets.
            loss.backward()  # Backpropagate gradients through the network.
            optimizer.step()  # Update model parameters using the computed gradients.
            epoch_loss += loss.item() * contexts.size(0)  # Accumulate loss scaled by batch size.
        average_loss = epoch_loss / len(dataset)  # Compute the average loss over the dataset.
        if (epoch + 1) % 10 == 0 or epoch == 0:  # Log progress periodically and for the first epoch.
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")  # Output training status for monitoring.
    prompt = "transformers excel at"  # Define a prompt for qualitative evaluation.
    predicted_word = generate_next_word(model, prompt, vocab, context_length, device)  # Generate the next word after the prompt.
    print(f"Prompt: '{prompt}' -> Predicted next word: '{predicted_word}'")  # Display the prediction result for reference.
if __name__ == "__main__":  # Execute the training routine when the script is run directly.
    train_model()  # Run the training and evaluation process.
