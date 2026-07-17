import torch
import torch.nn as nn
from positional import RotaryPositionalEncoding
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size, d_model, heads, d_ff, num_enc_layers, num_dec_layers, p, pos_method) :

        super().__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(trg_vocab_size, d_model)

        self.encoder_layers = nn.ModuleList([Encoder(d_model, heads, d_ff, p) for _ in range(num_enc_layers)])
        self.decoder_layers  = nn.ModuleList([Decoder(d_model, heads, d_ff, p) for _ in range(num_dec_layers)])

        self.fc_out = nn.Linear(d_model, trg_vocab_size) # Unembedding layer
        self.dropout = nn.Dropout(p)

        self.pos_method = pos_method

    def generate_mask(self, src, trg, pad_idx):

        trg_len = trg.size(1)

        # src mask is for masking pad tokens in the tokens. Used in encoder self attention and cross attention ()
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2) # [N, 1, 1, src_len] - Masks out pad tokens across all queries and heads
        trg_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2) # [N, 1, 1, trg_len] - Masks out pad tokens across all queries and heads

        nopeak_mask = (1 - torch.triu(torch.ones(1, trg_len, trg_len), diagonal=1)).bool().to(trg.device) # Lower triangular (excluding diagonal) (since you cant attend to current or future tokens) - [1, trg_len, trg_len]
        trg_mask = trg_mask & nopeak_mask # nopeak_mask is broadcasted across all batches - final shape = [N, 1, trg_len, trg_len] where future tokens and pad tokens are masked. Used in decoder self attention. Masks on all heads

        return src_mask, trg_mask

    def forward(self, src, trg, pad_idx):

        # src = [N, src_len] , trg = [N, tgt_len]

        src_mask, trg_mask = self.generate_mask(src, trg, pad_idx)

        src_embedding = self.dropout(self.encoder_embedding(src)) # [N, src_len, d_model]
        trg_embedding = self.dropout(self.decoder_embedding(trg)) # [N, trg_len, d_model]

        enc_output = src_embedding 
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output, src_mask, self.pos_method)

        # encoder output size = [N, seq_len, d_model] which will be used as K and V values for the cross attention

        dec_output = trg_embedding
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer(dec_output, enc_output, src_mask, trg_mask, self.pos_method) 
        
        # decoder output = [N, tgt_len, d_model]
        
        output = self.fc_out(dec_output) # [N, tgt_len, trg_vocab_size] - At every timestep, we get the predicted scores/logits.

        return output
