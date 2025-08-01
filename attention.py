import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(Q, K, V, mask=None):
    d_k = Q.size[1]
    
    similarity_mat = Q@K.T
    if mask:
        similarity_mat[:, mask] = -torch.math.inf
    probability_mat = F.softmax(similarity_mat/torch.math.sqrt(d_k))
    new_embeddings = probability_mat@V
    return new_embeddings
    

""" This is a slow implementation using for loops """
class Multihead_Attention:
    """
    Implementation of multi-head attention as specified by the paper Attention Is All You Need
    https://arxiv.org/pdf/1706.03762
    
    Parameters
    ------------
    h: int
        Number of heads
    d_model: int
        Dimension of inputs keys and queries
    d_k: int
        Dimension of learned linear projection
    d_v: int
        Output dimension
    
    Attributes
    ------------
    W_Q: torch.Tensor
        Tensor holding h different weight matrices associated with the queries
    W_K: torch.Tensor
        Tensor holding h different weight matrices associated with the keys
    W_V: torch.Tensor
        Tensor holding h different weight matrices associated with the values
    W_O: torch.Tensor
        Tensor holding weight matrix for final output transformation
    """
    def __init__(self, h, d_model, d_k, d_v, mask=None):
        self.num_heads = h
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.mask = mask
        
        self.W_Q = torch.rand(h, d_model, d_k)
        self.W_K = torch.rand(h, d_model, d_k)
        self.W_V = torch.rand(h, d_model, d_v)
        self.W_O = torch.rand(h*d_v, d_model)
        
    """
    Forward pass of multi-head attention as specified by the paper Attention Is All You Need
    https://arxiv.org/pdf/1706.03762
    
    Parameters
    ------------
    Q: Torch.Tensor
        Matrix of Queries
    K: Torch.Tensor
        Matrix of Keys
    V: Torch.Tensor
        Value matrix
    mask: Default None,  Torch.Tensor
        Boolean matrix used to modify Q@K.T before softmax
    """
    def forward(self, Q, K, V):
        heads = [attention(Q*self.W_Q[i], K*self.W_K[i], V*self.W_V[i], self.mask) for i in range(self.num_heads)]
        concat = torch.concat(heads)
        result = concat@self.W_O
        return result