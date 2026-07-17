import torch
import torch.nn as nn
import math

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, head_dim, max_seq_len=256):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        position = torch.arange(max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        self.register_buffer("sin", torch.sin(sinusoid_inp))
        self.register_buffer("cos", torch.cos(sinusoid_inp))

    def forward(self, x):
        # x: [batch, heads, seq_len, head_dim]
        seq_len = x.size(2)
        sin = self.sin[:seq_len, :]
        cos = self.cos[:seq_len, :]
        # Expand sin/cos to match batch and heads
        while sin.dim() < x.dim():
            sin = sin.unsqueeze(0)
            cos = cos.unsqueeze(0)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot
    
class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_seq_len=256):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.relative_attention_bias = nn.Embedding(2 * max_seq_len - 1, num_heads)

    def forward(self, q_len, k_len):
        device = self.relative_attention_bias.weight.device
        context_position = torch.arange(q_len, dtype=torch.long, device=device)[:, None]
        memory_position  = torch.arange(k_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position + self.max_seq_len - 1
        bias = self.relative_attention_bias(relative_position)
        return bias.permute(2, 0, 1)  # [num_heads, q_len, k_len]