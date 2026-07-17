import torch.nn as nn
from mha import MHA
from ffn import FeedForward
            
class Decoder(nn.Module):

    def __init__(self, d_model, heads, d_ff, p):

        super().__init__()
        self.self_attention = MHA(d_model,heads)
        self.cross_attention = MHA(d_model,heads)
        self.feedforward = FeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x, enc_output, src_mask, tgt_mask, pos_method):

        # x = [N, tgt_len, d_model]
        # enc_output = [N, src_len, d_model]

        self_attn = self.self_attention(x,x,x,pos_method,tgt_mask) # Mutlihead self attention in decoder (masked future tokens)
        x = self.norm1(x+self.dropout(self_attn)) # Add, Norm, Dropout

        cross_attn = self.cross_attention(x,enc_output,enc_output,pos_method,src_mask) # Multihead cross attention between encoder and decoder
        x = self.norm2(x+self.dropout(cross_attn)) # Add, Norm, Dropout

        ff_output = self.feedforward(x)
        x = self.norm3(x+self.dropout(ff_output))

        return x # Input size is preserved
