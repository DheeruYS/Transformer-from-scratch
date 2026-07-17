import torch.nn as nn
from mha import MHA
from ffn import FeedForward

class Encoder(nn.Module):

    def __init__(self, d_model, heads, d_ff, p):

        super().__init__()
        self.attention = MHA(d_model,heads)
        self.feedforward = FeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x, mask, pos_method):

        # x size = [N, seq_len, d_model] , same size is preserved after every layer

        attention_output = self.attention(x,x,x,pos_method,mask) # Mutlihead attention layer
        x = self.norm1(x+self.dropout(attention_output)) # Add and norm
        
        ff_output = self.feedforward(x) # Position independent FFN
        x = self.norm2(x+self.dropout(ff_output)) # Add and norm

        return x # After 1 layer of encoder block, same size is maintained