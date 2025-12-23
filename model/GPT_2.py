import torch
import torch.nn as nn
from model.Transformer import Transformer
GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 256, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}


class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks  =nn.ModuleList([
            Transformer(cfg) for _ in range(cfg["n_layers"])
        ])
        self.layer_norm = nn.LayerNorm(cfg["emb_dim"])
        self.proj = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self, x):
        batch_size, sqe_len = x.shape
        token_emd = self.token_embedding(x)
        position_emd = self.position_embedding(torch.arange(0, sqe_len, device=x.device))
        x = token_emd + position_emd
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = self.proj(x)
        return x
