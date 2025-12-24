import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert d_out %  num_heads == 0, "d_out must be divisible by num_heads"
        self.d_in = d_in
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        keys = self.W_key(x) # batch_size, num_tokens, d_out
        queries = self.W_query(x) # batch_size, num_tokens, d_out
        values = self.W_value(x) # batch_size, num_tokens, d_out

        # 切分为多头
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim) # batch_size, num_tokens, num_heads, head_dim
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)  # (b,num_heads,num_tokens,head_dim)
        queries = queries.transpose(1, 2)  # (b,num_heads,num_tokens,head_dim)
        values = values.transpose(1, 2)  # (b,num_heads,num_tokens,head_dim)

        # attention_scores
        attention_scores = queries @ keys.transpose(2,3) # batch_size, num_heads, num_tokens, num_tokens

        # masked fill
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(mask_bool, -float('inf'))

        # attention_weight
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1) # batch_size, num_heads, num_tokens, num_tokens
        attention_weights = self.dropout(attention_weights)
        # context_vec
        context_vec = (attention_weights @ values).transpose(1,2) # batch_size, num_tokens, num_heads, head_dim
        context_vec = context_vec.contiguous().view(
            batch_size, num_tokens, self.d_out
        )

        output = self.out_proj(context_vec)
        return output

# inputs = torch.tensor(
# [[0.43, 0.15, 0.89], # Your (x^1)
# [0.55, 0.87, 0.66], # journey (x^2)
# [0.57, 0.85, 0.64], # starts (x^3)
# [0.22, 0.58, 0.33], # with (x^4)
# [0.77, 0.25, 0.10], # one (x^5)
# [0.05, 0.80, 0.55]] # step (x^6)
# )
# batch = torch.stack((inputs, inputs), dim=0)
# torch.manual_seed(123)
# batch_size, context_length, d_in = batch.shape
# d_out = 2
# mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
# context_vecs = mha(batch)
# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)shape