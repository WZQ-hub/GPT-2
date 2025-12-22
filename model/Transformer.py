import torch
import torch.nn as nn
from model.MultiHeadAttention import MultiHeadAttention

GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ff1 = nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
        self.GELU = nn.GELU()
        self.ff2 = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])



    def forward(self, x):
        x = self.ff1(x)
        x = self.GELU(x)
        x = self.ff2(x)
        return x

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(cfg["emb_dim"])
        self.m_attention = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            dropout = cfg["drop_rate"],
            num_heads = cfg["n_heads"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):
        shortcut = x
        x = self.LayerNorm(x)
        x = self.m_attention(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x

        x = self.LayerNorm(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut

        return x



