# %%
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# %%
class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = key.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / np.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  
        attention = F.softmax(scores, dim=-1)
        out = attention.matmul(value)
        return out, attention

y = torch.rand(1, 28, 28)
out = ScaledDotProductAttention()(y, y, y)
out[0].shape, out[1].shape

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, features, num_heads, bias=True, activation=F.relu_):
        super(MultiHeadAttention, self).__init__()
        assert features % num_heads == 0, f'"features"(features) should be divisible by "head_num"(num_heads)'
        
        self.features = features
        self.num_heads = num_heads
        self.bias = bias
        self.depth = features // num_heads
        self.act = activation

        self.wq = nn.Linear(features, features, bias=bias)
        self.wk = nn.Linear(features, features, bias=bias)
        self.wv = nn.Linear(features, features, bias=bias)

        self.fc = nn.Linear(features, features, bias=bias)
        
    def split_heads(self, x, batch_size):
        # batch_size, num_heads, seq_len, depth
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute([0, 2, 1, 3])
    
    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch_size, seq_len, features)
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        # print(q.shape, k.shape, v.shape)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # print(q.shape, k.shape, v.shape)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = ScaledDotProductAttention()(q, k, v, mask)
        # print(scaled_attention.shape, attention_weights.shape)

        scaled_attention = scaled_attention.permute([0, 2, 1, 3])

        concat_attention = scaled_attention.reshape(batch_size, -1, self.features)

        out = self.fc(concat_attention)
        if self.act is not None:
            out = self.act(out)

        return out, attention_weights # (batch_size, seq_len_q, features), (batch_size, num_head, seq_len_q, seq_len_k)

temp_mha = MultiHeadAttention(features=28, num_heads=4)
out, attn = temp_mha(q=torch.rand(1, 28, 28), k=y, v=y, mask=None)
print(out.shape, attn.shape)

# %%
class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, mlp_dim, out_features=None, rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        actual_out_dim = in_features if out_features is None else out_features
        layer_list = [
            nn.Linear(in_features, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, actual_out_dim),
            nn.Dropout(rate)
        ]

        self.net = nn.Sequential(*layer_list)
    def forward(self, x):
        return self.net(x)

sample_ffn = FeedForwardNetwork(512, 2048)
sample_ffn(torch.rand(64, 50, 512)).shape

# %%
class Encoder1DBlock(nn.Module):
    def __init__(self, in_features, num_heads, mlp_dim, rate):
        super(Encoder1DBlock, self).__init__()
        self.ln1 = nn.LayerNorm(in_features)

        self.attn = MultiHeadAttention(in_features, num_heads)

        self.dropout = nn.Dropout(rate)

        self.ln2 = nn.LayerNorm(in_features)
        self.ffn = FeedForwardNetwork(in_features, mlp_dim, rate=rate)

    def forward(self, x):
        out_1 = self.ln1(x)
        out_1, _ = self.attn(out_1, out_1, out_1)
        out_1 = self.dropout(out_1)
        out_1 = x + out_1

        out_2 = self.ln2(out_1)
        out_2 = self.ffn(out_2)

        return out_1 + out_2

sample_EB = Encoder1DBlock(784, 7, 1024, 0.1)
sample_EB(torch.rand(16, 50, 784))
# %%
class Encoder(nn.Module):
    def __init__(self, in_feature, num_layers, num_heads, mlp_dim, rate=0.1):
        super(Encoder, self).__init__()
        # self.pe = ?

        self.dropout = nn.Dropout(rate)

        self.encoder  = nn.Sequential(*[Encoder1DBlock(in_feature, num_heads, mlp_dim, rate) for i in range(num_layers)])

        self.ln = nn.LayerNorm(in_feature)

    def forward(self, x):
        # out = self.pe(x)
        out = self.dropout(x)
        out = self.encoder(out)
        out = self.ln(out)
        return out

sample_En = Encoder(784, 3, 7, 1024)
sample_En(torch.rand(16, 5, 784)).shape

# %%
