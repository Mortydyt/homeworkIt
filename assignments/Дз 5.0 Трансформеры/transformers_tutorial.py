"""
Comprehensive Transformers Tutorial
Complete implementation covering all 7 days of Transformer architecture understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


# Day 1: Architecture and Tokenization
class TokenizerDemo:
    """Demonstrates different tokenization approaches (Day 1)"""

    def __init__(self):
        self.vocab = {}
        self.word_to_id = {}
        self.id_to_word = {}

    def simple_tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization"""
        return text.lower().split()

    def character_tokenize(self, text: str) -> List[str]:
        """Character-level tokenization"""
        return list(text.lower())

    def subword_tokenize_demo(self, text: str) -> List[str]:
        """Demo of subword tokenization (simplified BPE-like)"""
        # Simple subword approach - split on common patterns
        words = text.lower().split()
        subwords = []
        for word in words:
            if len(word) <= 3:
                subwords.append(word)
            else:
                # Simple subword splitting
                subwords.append(word[:2])
                subwords.append(word[2:])
        return subwords

    def build_vocab(self, texts: List[str]) -> dict:
        """Build vocabulary from texts"""
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.simple_tokenize(text))

        # Count frequencies
        from collections import Counter
        token_counts = Counter(all_tokens)

        # Create vocabulary
        vocab = {'<pad>': 0, '<unk>': 1}
        for i, (token, count) in enumerate(token_counts.most_common(1000), 2):
            vocab[token] = i

        self.vocab = vocab
        self.word_to_id = vocab
        self.id_to_word = {v: k for k, v in vocab.items()}
        return vocab

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.simple_tokenize(text)
        return [self.word_to_id.get(token, 1) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = [self.id_to_word.get(tid, '<unk>') for tid in token_ids]
        return ' '.join(tokens)

    def explain_tokenization(self, text: str):
        """Explain tokenization process (Day 1, Task 3)"""
        print("🔤 Tokenization Explanation")
        print("=" * 40)
        print(f"Original text: '{text}'")
        print()

        # Word-level tokenization
        word_tokens = self.simple_tokenize(text)
        print(f"1. Word-level tokens: {word_tokens}")
        print(f"   Count: {len(word_tokens)} tokens")

        # Character-level tokenization
        char_tokens = self.character_tokenize(text)
        print(f"2. Character-level tokens: {char_tokens}")
        print(f"   Count: {len(char_tokens)} tokens")

        # Subword tokenization
        subword_tokens = self.subword_tokenize_demo(text)
        print(f"3. Subword tokens: {subword_tokens}")
        print(f"   Count: {len(subword_tokens)} tokens")

        # Vocabulary encoding
        if self.vocab:
            encoded = self.encode(text)
            decoded = self.decode(encoded)
            print(f"4. Encoded IDs: {encoded}")
            print(f"5. Decoded: '{decoded}'")


# Day 2: Self-Attention Mechanism
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention (Day 2)"""

    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: Query tensor [batch_size, seq_len, d_k]
            K: Key tensor [batch_size, seq_len, d_k]
            V: Value tensor [batch_size, seq_len, d_v]
            mask: Optional mask [batch_size, seq_len, seq_len]

        Returns:
            output: Weighted sum of values
            attention_weights: Attention weights
        """
        d_k = Q.size(-1)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (Day 2)"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # [batch_size, seq_len, d_model]
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # Apply attention
        attention_output, _ = self.attention(Q, K, V, mask)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final linear projection
        output = self.W_o(attention_output)

        return output


# Day 3: Positional Encoding
class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformers (Day 3)"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate division term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        # Apply sin to even positions, cos to odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [seq_len, batch_size, d_model]

        Returns:
            x + positional encoding
        """
        return x + self.pe[:x.size(0), :]


# Day 4: Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network (Day 4)"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # First linear transformation + activation
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Second linear transformation
        x = self.w2(x)

        return x


# Day 5: Encoder Layer
class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer (Day 5)"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


# Day 6: Transformer Encoder
class TransformerEncoder(nn.Module):
    """Complete Transformer Encoder (Day 6)"""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input token IDs [batch_size, seq_len]
            mask: Optional attention mask

        Returns:
            Encoded representations [batch_size, seq_len, d_model]
        """
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return x


# Day 7: Complete Transformer Model
class TransformerClassifier(nn.Module):
    """Transformer-based text classifier (Day 7)"""

    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 512,
                 num_heads: int = 8, num_layers: int = 6, d_ff: int = 2048,
                 max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input token IDs [batch_size, seq_len]
            mask: Optional attention mask

        Returns:
            Class logits [batch_size, num_classes]
        """
        # Encode input
        encoded = self.encoder(x, mask)  # [batch_size, seq_len, d_model]

        # Global average pooling over sequence length
        pooled = encoded.mean(dim=1)  # [batch_size, d_model]

        # Classify
        logits = self.classifier(pooled)  # [batch_size, num_classes]

        return logits

    def generate_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """Generate attention weights for visualization"""
        # This would require modifying the encoder to expose attention weights
        # Simplified version for demonstration
        return torch.ones(x.size(0), 8, x.size(1), x.size(1))


# Visualization and Demo Functions
def create_attention_heatmap(attention_weights: torch.Tensor, tokens: List[str],
                            title: str = "Attention Weights"):
    """Create heatmap visualization of attention weights"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights[0, 0].detach().cpu().numpy(),
                xticklabels=tokens, yticklabels=tokens, cmap='Blues')
    plt.title(title)
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.tight_layout()
    return plt


def demonstrate_transformer_architecture():
    """Comprehensive demonstration of all components"""
    print("🤖 Complete Transformer Architecture Demo")
    print("=" * 50)

    # Day 1: Tokenization Demo
    print("\n📅 Day 1: Tokenization")
    tokenizer = TokenizerDemo()
    sample_texts = [
        "Hello world! How are you?",
        "Transformers are amazing models.",
        "Natural language processing is fun."
    ]
    tokenizer.build_vocab(sample_texts)
    tokenizer.explain_tokenization("Hello, transformers!")

    # Day 2: Self-Attention Demo
    print("\n📅 Day 2: Self-Attention Mechanism")
    d_model, num_heads = 512, 8
    batch_size, seq_len = 2, 10

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    attention = MultiHeadAttention(d_model, num_heads)
    attn_output = attention(x, x, x)
    print(f"Input shape: {x.shape}")
    print(f"Attention output shape: {attn_output.shape}")

    # Day 3: Positional Encoding Demo
    print("\n📅 Day 3: Positional Encoding")
    pos_encoder = PositionalEncoding(d_model)
    x_seq = torch.randn(seq_len, batch_size, d_model)
    x_encoded = pos_encoder(x_seq)
    print(f"Sequence shape: {x_seq.shape}")
    print(f"With positional encoding: {x_encoded.shape}")

    # Day 4: Feed-Forward Demo
    print("\n📅 Day 4: Feed-Forward Network")
    ff = PositionwiseFeedForward(d_model, 2048)
    ff_output = ff(x)
    print(f"Feed-forward output shape: {ff_output.shape}")

    # Day 5: Encoder Layer Demo
    print("\n📅 Day 5: Encoder Layer")
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, 2048)
    layer_output = encoder_layer(x)
    print(f"Encoder layer output shape: {layer_output.shape}")

    # Day 6: Complete Encoder Demo
    print("\n📅 Day 6: Complete Transformer Encoder")
    vocab_size = 1000
    encoder = TransformerEncoder(vocab_size, d_model, num_heads, 6, 2048)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    encoded_output = encoder(token_ids)
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Encoded output shape: {encoded_output.shape}")

    # Day 7: Complete Model Demo
    print("\n📅 Day 7: Complete Transformer Classifier")
    num_classes = 3
    classifier = TransformerClassifier(vocab_size, num_classes, d_model, num_heads)
    logits = classifier(token_ids)
    probabilities = F.softmax(logits, dim=-1)
    print(f"Classifier logits shape: {logits.shape}")
    print(f"Class probabilities shape: {probabilities.shape}")
    print(f"Sample probabilities: {probabilities[0]}")

    print("\n🎉 Transformer Architecture Demo Completed!")


def attention_mechanism_explanation():
    """Detailed explanation of attention mechanism"""
    print("🧠 Understanding Self-Attention Mechanism")
    print("=" * 50)

    # Create example matrices
    seq_len, d_k = 4, 3

    # Example sequence: ["The", "cat", "sat", "on"]
    # Create random Q, K, V matrices for demonstration
    Q = torch.tensor([[1.0, 0.0, 0.0],  # "The" query
                      [0.0, 1.0, 0.0],  # "cat" query
                      [0.0, 0.0, 1.0],  # "sat" query
                      [0.5, 0.5, 0.0]]) # "on" query

    K = torch.tensor([[1.0, 0.0, 0.0],  # "The" key
                      [0.0, 1.0, 0.0],  # "cat" key
                      [0.0, 0.0, 1.0],  # "sat" key
                      [0.5, 0.5, 0.0]]) # "on" key

    V = torch.tensor([[0.1, 0.2],      # "The" value
                      [0.3, 0.4],      # "cat" value
                      [0.5, 0.6],      # "sat" value
                      [0.7, 0.8]])     # "on" value

    tokens = ["The", "cat", "sat", "on"]
    print(f"Input tokens: {tokens}")
    print(f"Q matrix:\n{Q}")
    print(f"K matrix:\n{K}")
    print(f"V matrix:\n{V}")

    # Calculate attention scores
    attention = ScaledDotProductAttention()
    output, attn_weights = attention(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))

    print(f"\nAttention weights:\n{attn_weights[0]}")
    print(f"Output:\n{output[0]}")

    print("\n💡 Key Insights:")
    print("1. Q (Query) represents current token's 'question'")
    print("2. K (Key) represents all tokens' 'keys' to match against")
    print("3. V (Value) represents the actual information to extract")
    print("4. Attention weights show how much each token attends to others")
    print("5. Output is weighted sum of values based on attention weights")


def main():
    """Main function to run all demonstrations"""
    try:
        # Run comprehensive demo
        demonstrate_transformer_architecture()

        # Run attention mechanism explanation
        attention_mechanism_explanation()

        print("\n🎓 Learning Summary:")
        print("✅ Day 1: Tokenization and vocabulary")
        print("✅ Day 2: Self-attention and multi-head attention")
        print("✅ Day 3: Positional encoding")
        print("✅ Day 4: Position-wise feed-forward networks")
        print("✅ Day 5: Transformer encoder layer")
        print("✅ Day 6: Complete transformer encoder")
        print("✅ Day 7: End-to-end transformer model")

        print("\n🚀 You now have a complete understanding of Transformer architecture!")

    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")


if __name__ == "__main__":
    main()