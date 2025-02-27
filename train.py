import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import random
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import re
import math

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Improved tokenizer
def tokenize(text):
    # Convert to lowercase
    text = text.lower()
    # Replace URLs with token
    text = re.sub(r'https?://\S+', '<url>', text)
    # Replace numbers with token
    text = re.sub(r'\d+', '<num>', text)
    # Replace punctuation with space
    text = re.sub(r'[^\w\s<>]', ' ', text)
    # Split by whitespace and filter empty tokens
    tokens = [t for t in text.split() if t]
    return tokens

# Text augmentation functions
def random_swap(tokens, swap_prob=0.05):
    """Randomly swap adjacent tokens"""
    if len(tokens) < 2:
        return tokens
    
    new_tokens = tokens.copy()
    for i in range(len(tokens) - 1):
        if random.random() < swap_prob:
            new_tokens[i], new_tokens[i+1] = new_tokens[i+1], new_tokens[i]
    
    return new_tokens

def random_dropout(tokens, dropout_prob=0.05):
    """Randomly drop tokens"""
    if len(tokens) < 3:  # Keep very short texts intact
        return tokens
    
    return [t for t in tokens if random.random() > dropout_prob]

def augment_text(text, augment_prob=0.3):
    """Apply augmentation with some probability"""
    if random.random() > augment_prob:
        return text
    
    tokens = tokenize(text)
    
    # Apply one of the augmentations randomly
    aug_type = random.choice(['swap', 'dropout'])
    if aug_type == 'swap':
        tokens = random_swap(tokens)
    else:
        tokens = random_dropout(tokens)
    
    return ' '.join(tokens)

class Vocabulary:
    def __init__(self, specials=None):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        self.specials = specials or ["<unk>", "<pad>", "<url>", "<num>"]
        
        # Add special tokens
        for token in self.specials:
            self.add_word(token)
        
        self.default_idx = self.word2idx.get("<unk>", 0)
        self.pad_idx = self.word2idx.get("<pad>", 0)
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word
        self.word_count[word] += 1
    
    def build_vocab(self, texts, min_freq=3):
        # Count all words
        for text in texts:
            for word in tokenize(text):
                self.word_count[word] += 1
        
        # Add frequent words to vocabulary
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                self.add_word(word)
        
        print(f"Vocabulary built with {len(self.word2idx)} words")
        
        # Get word frequencies for embedding weights initialization
        self.frequencies = []
        for i in range(len(self.word2idx)):
            word = self.idx2word.get(i)
            if word:
                self.frequencies.append(self.word_count[word])
            else:
                self.frequencies.append(0)
    
    def __call__(self, tokens):
        if isinstance(tokens, str):
            tokens = tokenize(tokens)
        
        return [self.word2idx.get(token, self.default_idx) for token in tokens]
    
    def __len__(self):
        return len(self.word2idx)

class AGNewsDataset(Dataset):
    def __init__(self, csv_path, vocab=None, max_length=100, augment=False):
        self.data = []
        self.max_length = max_length
        self.augment = augment
        
        # Load data from CSV
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    if len(row) >= 2:
                        try:
                            label = int(row[0])  # Labels in AG News are 1, 2, 3, 4
                            text = row[1]
                            if len(row) > 2:
                                text += ' ' + row[2]
                            self.data.append((label, text))
                        except (ValueError, IndexError) as e:
                            print(f"Skipping row due to error: {e}, Row: {row}")
                            continue
        except Exception as e:
            print(f"Error loading dataset from {csv_path}: {e}")
            raise
            
        print(f"Loaded {len(self.data)} examples from {csv_path}")
        
        if len(self.data) == 0:
            raise ValueError(f"No valid data loaded from {csv_path}")
            
        if vocab is None:
            self.vocab = Vocabulary()
            print("Building vocabulary...")
            self.vocab.build_vocab([text for _, text in self.data])
        else:
            self.vocab = vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, text = self.data[idx]
        
        # Apply augmentation during training
        if self.augment and random.random() < 0.3:
            text = augment_text(text)
            
        return label - 1, text  # Convert to 0-based indexing

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    
    for label, text in batch:
        label_list.append(label)
        tokens = tokenize(text)
        # Truncate if longer than max_length
        if len(tokens) > 100:
            tokens = tokens[:100]
        lengths.append(len(tokens))
        token_ids = vocab(tokens)
        text_list.append(torch.tensor(token_ids, dtype=torch.int64))
    
    # Pad sequences - only to the max length in this batch
    max_length = max(lengths)
    padded_texts = []
    for text_tensor in text_list:
        padded = torch.full((max_length,), vocab.pad_idx, dtype=torch.int64)
        padded[:len(text_tensor)] = text_tensor
        padded_texts.append(padded)
    
    return (
        torch.tensor(label_list, dtype=torch.int64).to(device),
        torch.stack(padded_texts).to(device),
        torch.tensor(lengths, dtype=torch.int64).to(device)
    )

class EnhancedTextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, 
                 filter_sizes=[2, 3, 4, 5], num_filters=128, dropout=0.5, 
                 vocab_frequencies=None):
        super(EnhancedTextCNN, self).__init__()
        
        # Initialize embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if vocab_frequencies:
            self._init_embedding_weights(vocab_frequencies, embed_dim)
            
        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, 
                      out_channels=num_filters, 
                      kernel_size=fs,
                      padding=fs // 2)  # Added padding for better feature capture
            for fs in filter_sizes
        ])
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in filter_sizes
        ])
        
        # Dropout and activation
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 1.2)  # Increased dropout for the classifier
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)  # Added LeakyReLU for more stable gradients
        
        # Hidden layers
        total_filter_size = len(filter_sizes) * num_filters
        self.fc1 = nn.Linear(total_filter_size, total_filter_size // 2)
        self.bn_fc = nn.BatchNorm1d(total_filter_size // 2)  # Batch norm for fc layer
        self.fc2 = nn.Linear(total_filter_size // 2, num_class)
        
    def _init_embedding_weights(self, frequencies, embed_dim):
        """Improved embedding initialization"""
        weights = self.embedding.weight.data
        for i, freq in enumerate(frequencies):
            if freq > 0:
                # Use Xavier/Glorot initialization with frequency scaling
                scale = 0.08 / (math.sqrt(freq) + 1)
                weights[i].uniform_(-scale, scale)
    
    def forward(self, text, lengths=None):
        # text: [batch_size, seq_length]
        embedded = self.embedding(text)  # [batch_size, seq_length, embed_dim]
        
        # Apply embedding dropout
        embedded = self.dropout1(embedded)
        
        # Transpose for conv1d which expects [batch, embed_dim, seq_len]
        embedded = embedded.transpose(1, 2)  # [batch_size, embed_dim, seq_length]
        
        # Apply convolutions, batch norm, and activation
        conv_results = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            conved = self.leaky_relu(bn(conv(embedded)))
            # Global max pooling
            pooled = nn.functional.adaptive_max_pool1d(conved, 1).squeeze(2)
            conv_results.append(pooled)
        
        # Concatenate pooled features
        cat = torch.cat(conv_results, dim=1)
        
        # Apply dropout and hidden layer with batch norm
        dropped = self.dropout2(cat)
        hidden = self.leaky_relu(self.bn_fc(self.fc1(dropped)))
        
        return self.fc2(hidden)

def calculate_class_weights(dataset):
    """Calculate class weights inversely proportional to class frequency"""
    labels = [label for label, _ in dataset.data]
    class_counts = Counter([l-1 for l in labels])  # Convert to 0-based indexing
    
    # Get counts for each class
    counts = torch.zeros(4)
    for i in range(4):
        counts[i] = class_counts.get(i, 0)
    
    # Calculate weights inversely proportional to count
    weights = 1.0 / counts
    # Normalize to sum to len(classes)
    weights = weights * (len(counts) / weights.sum())
    
    # Give extra weight to the World class (index 0)
    weights[0] *= 1.2
    
    print(f"Class weights: {weights}")
    return weights

def train_model(train_path, test_path, epochs=15, batch_size=128, 
                eval_every=100, early_stopping_patience=8):
    global vocab  # Make vocab available to collate function
    
    # Create datasets
    train_dataset = AGNewsDataset(train_path, augment=True)  # Enable augmentation
    vocab = train_dataset.vocab
    test_dataset = AGNewsDataset(test_path, vocab=vocab, augment=False)
    
    # Split training data into train and validation
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_batch
    )
    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_batch
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_batch
    )
    
    # Create model with improved architecture
    num_class = 4  # AG News has 4 classes
    vocab_size = len(vocab)
    embed_dim = 200  # Increased for better representation
    
    model = EnhancedTextCNN(
        vocab_size=vocab_size, 
        embed_dim=embed_dim, 
        num_class=num_class,
        num_filters=128,
        dropout=0.5,
        vocab_frequencies=vocab.frequencies if hasattr(vocab, 'frequencies') else None
    ).to(device)
    
    # Calculate class weights for weighted loss
    class_weights = calculate_class_weights(train_dataset).to(device)
    
    # Define loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    
    # Learning rate scheduler (gentler curve)
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0008,
        total_steps=total_steps,
        pct_start=0.1,  # Warmup for 10% of training (increased from 5%)
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    # Metrics tracking
    best_val_acc = 0.0
    early_stopping_counter = 0
    global_step = 0
    
    # Set a minimum improvement threshold to avoid early stopping on small fluctuations
    min_improvement = 0.0005
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        batch_count = 0
        
        for idx, (labels, texts, lengths) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(texts, lengths)
            loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            epoch_loss += loss.item()
            accuracy = (predictions.argmax(1) == labels).float().mean()
            epoch_acc += accuracy.item()
            batch_count += 1
            global_step += 1
            
            # Print progress
            if idx % 200 == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Batch: {idx}/{len(train_dataloader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {accuracy.item():.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Validation during training
            if global_step % eval_every == 0:
                model.eval()
                val_loss = 0
                val_acc = 0
                val_count = 0
                
                with torch.no_grad():
                    for val_labels, val_texts, val_lengths in val_dataloader:
                        val_preds = model(val_texts, val_lengths)
                        val_batch_loss = criterion(val_preds, val_labels)
                        val_batch_acc = (val_preds.argmax(1) == val_labels).float().mean()
                        
                        val_loss += val_batch_loss.item()
                        val_acc += val_batch_acc.item()
                        val_count += 1
                
                avg_val_loss = val_loss / val_count
                avg_val_acc = val_acc / val_count
                
                print(f"\nStep {global_step}: Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}\n")
                
                # Save model based on validation accuracy (not loss)
                if avg_val_acc > best_val_acc + min_improvement:
                    best_val_acc = avg_val_acc
                    torch.save(model.state_dict(), "best_val_acc_model.pth")
                    print(f"New best validation accuracy: {best_val_acc:.4f}")
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                # Early stopping based on validation accuracy
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {early_stopping_counter} evaluations without improvement")
                    break
                
                # Back to training mode
                model.train()
        
        # Calculate epoch averages
        avg_loss = epoch_loss / batch_count
        avg_acc = epoch_acc / batch_count
        
        # Evaluation on test set after each epoch
        model.eval()
        test_loss = 0
        test_acc = 0
        test_count = 0
        
        with torch.no_grad():
            for labels, texts, lengths in test_dataloader:
                predictions = model(texts, lengths)
                loss = criterion(predictions, labels)
                
                test_loss += loss.item()
                test_acc += (predictions.argmax(1) == labels).float().mean().item()
                test_count += 1
        
        avg_test_loss = test_loss / test_count
        avg_test_acc = test_acc / test_count
        
        # Print epoch summary
        time_elapsed = time.time() - start_time
        print(f"\nEpoch: {epoch+1}/{epochs} completed in {time_elapsed:.0f}s")
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}")
        print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
        
        # Check for early stopping
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered. Training ended.")
            break
    
    # Final evaluation with best validation accuracy model
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load("best_val_acc_model.pth"))
    model.eval()
    
    # Detailed evaluation on test set
    correct_by_class = [0] * num_class
    total_by_class = [0] * num_class
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for labels, texts, lengths in test_dataloader:
            predictions = model(texts, lengths)
            preds = predictions.argmax(1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Count correct predictions by class
            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                
                total_by_class[label] += 1
                if pred == label:
                    correct_by_class[label] += 1
    
    # Print class-wise accuracy
    print("\nClass-wise Accuracy:")
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    for i in range(num_class):
        acc = correct_by_class[i] / total_by_class[i] if total_by_class[i] > 0 else 0
        print(f"Class {i} ({class_names[i]}): {acc:.4f} ({correct_by_class[i]}/{total_by_class[i]})")
    
    # Print overall accuracy
    overall_acc = sum(correct_by_class) / sum(total_by_class)
    print(f"\nOverall Test Accuracy: {overall_acc:.4f}")
    
    # Calculate and print confusion matrix
    confusion_matrix = np.zeros((num_class, num_class), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion_matrix[label, pred] += 1
    
    print("\nConfusion Matrix:")
    print("Predicted →   |", end="")
    for i in range(num_class):
        print(f"  {class_names[i]:9s}|", end="")
    print("\nActual ↓       |" + "-" * (num_class * 13))
    
    for i in range(num_class):
        print(f"{class_names[i]:13s}|", end="")
        for j in range(num_class):
            print(f" {confusion_matrix[i, j]:9d} |", end="")
        print()
    
    # Save the final model with all necessary components
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "class_names": class_names,
        "accuracy": overall_acc,
        "confusion_matrix": confusion_matrix,
        "model_config": {
            "embed_dim": embed_dim,
            "num_filters": 128,
            "filter_sizes": [2, 3, 4, 5]
        }
    }, "ag_news_cnn_model.pth")
    
    print("Final model saved to ag_news_cnn_model.pth")
    
    return overall_acc, model

# Main execution
if __name__ == "__main__":
    # Define paths to the dataset files
    train_path = 'ag_news_csv/train.csv'
    test_path = 'ag_news_csv/test.csv'
    
    # Check if files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Dataset files not found. Please download the AG News dataset first.")
        exit(1)
    
    # Train the model with improved hyperparameters
    accuracy, model = train_model(
        train_path=train_path,
        test_path=test_path,
        epochs=15,           # Increased from 10
        batch_size=128,
        eval_every=200,      # Evaluate on validation set every 200 steps
        early_stopping_patience=8  # Increased from 5
    )
