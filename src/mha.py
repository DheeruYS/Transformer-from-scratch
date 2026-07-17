import torch
import torch.nn as nn
import math
from positional import RotaryPositionalEncoding
from positional import RelativePositionBias

class MHA(nn.Module):

    def __init__(self, d_model, heads):

        super().__init__()
        
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        self.rope = RotaryPositionalEncoding(self.head_dim)
        self.positional_bias = RelativePositionBias(self.heads)

        assert (self.head_dim * heads == d_model), "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.reshape(batch_size,seq_len,self.heads,self.head_dim).transpose(1,2)
    
    def scaled_dot_product_attention(self, Q, K, V, pos_method, mask=None):
        energy = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim) # [N, heads, query_len, head_dim] * [N, heads, head_dim, key_len] = [N, heads, query_len, key_len]
        # energy(q(i) k(j)) gives the energy score/logit for the similarity between ith query vector and jth key vector  

        if pos_method == "relative":
            bias = self.positional_bias(Q.shape[2], K.shape[2])  # [num_heads, q_len, k_len]
            energy = energy + bias.unsqueeze(0)  # [N, heads, q_len, k_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e20) # Huge negative value wherever mask value is null in the mask matrix, so that after softmax, a 0 is present in those positions.
            # broadcasts mask value to energy length by repeating the mask matrix

        attention = torch.softmax(energy, dim=-1) # For each query, we softmax the scores over all keys. For each query, how relevant (softmaxed) is the key
        output = torch.matmul(attention,V) # For each query, to get the attention vector, multiply the relevance of every key with the value vector
        return output # [N, heads, query_len, head_dim] - For every query, the attention vector of head_dim
    
    def combine_heads(self, x):
        batch_size, heads, query_len, head_dim = x.size()
        return x.transpose(1,2).reshape(batch_size, query_len, self.d_model) # Transpose since we need to combine the heads and head_dim, keeping them next to each other and then combining

    def forward(self, Q, K, V, pos_method, mask=None):

        # Q, K, V = [N, seq_len, d_model] with seq_len = query_len for Q and seq_len = key_len = value_len for K,V

        query_len, key_len, val_len = Q.shape[1], K.shape[1], V.shape[1] # No. of queries, no. of keys, no. of values. key_len = val_len always as both of them are pairs and come from the same source.
        # But, in cross attention, no. of queries might not match the no. of key value pairs.
        
        Q = self.split_heads(self.W_q(Q)) # [N, heads, query_len, head_dim] 
        K = self.split_heads(self.W_k(K)) # [N, heads, key_len, head_dim] 
        V = self.split_heads(self.W_v(V)) # [N, heads, val_len, head_dim]

        if pos_method == "rope":
            Q = self.rope(Q)
            K = self.rope(K)

        # Each head is processed independently and in parallel. Every head is a separate self attention mechanism which looks at all the tokens in the input, processes all queries.

        attn_output = self.scaled_dot_product_attention(Q,K,V,pos_method, mask) # Now for each head, for every query, it looks at all the tokens and gets an embedding of size head_dim
        # now, for all heads, we compute the embedding of size head_dim for every query vector looking at all the tokens. Now we concat all the embeddings in each query row, to get a combined embedding of size d_model
        # In this concatenated vector, there is no interaction between any heads. To share the information between the heads, it is sent through another linear layer, which preserves the size, but adds intra head information

        output = self.combine_heads(attn_output) # [N, query_len, d_model] - For every query, we have an embedding of original size (helps for add layer), but there is no interaction between heads still
        output = self.W_o(output) # [N, query_len, d_model] - Now, all the information from each head has been mixed appropriately

        return output
