# %%
import os, torch
import cv2 as cv
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, datasets, utils

# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
class ScaledDotProductAttention(nn.Module):
    def forward(self,Q,K,V,mask=None):
        d_K = K.size()[-1] # key dimension
        scores = Q.matmul(K.transpose(-2,-1)) / np.sqrt(d_K)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        attention = F.softmax(scores,dim=-1)
        out = attention.matmul(V)
        return out,attention

class MultiHeadedAttention(nn.Module):
    def __init__(self,d_feat=128, n_head=5, actv=F.relu, use_bias=True):

        super(MultiHeadedAttention, self).__init__()
        if (d_feat%n_head) != 0:
            raise ValueError("d_feat(%d) should be divisible by b_head(%d)"%(d_feat,n_head)) 
        self.d_feat = d_feat
        self.n_head = n_head
        self.d_head = self.d_feat // self.n_head
        self.actv = actv
        self.use_bias = use_bias
        
        self.SDPA = ScaledDotProductAttention()
        self.lin_Q = nn.Linear(self.d_feat,self.d_feat,self.use_bias)
        self.lin_K = nn.Linear(self.d_feat,self.d_feat,self.use_bias)
        self.lin_V = nn.Linear(self.d_feat,self.d_feat,self.use_bias)
        self.lin_O = nn.Linear(self.d_feat,self.d_feat,self.use_bias)

        self.dropout = nn.Dropout(p=self.dropout_rate)
    
    def forward(self,Q,K,V,mask=None):
        n_batch = Q.shape[0]
        Q_emb = self.lin_Q(Q) 
        K_emb = self.lin_K(K) 
        V_emb = self.lin_V(V)

        Q_emb = Q_emb.view(n_batch, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        K_emb = K_emb.view(n_batch, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        V_emb = V_emb.view(n_batch, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)

        out, attention = self.SDPA(Q_emb, K_emb, V_emb, mask)

        # Reshape x
        out = out.permute(0,2,1,3).contiguous()
        out = out.view(n_batch,-1,self.d_feat)

        # Linear
        out = self.lin_O(out)

        return out, attention

class EncoderLyaer(nn.Module):
    def __init__(self, d_feat=128, n_head=5, actv=F.relu, use_bias=True, features=256, rate=0.1):
        super(EncoderLyaer, self).__init__()
        self.d_feat = d_feat
        self.n_head = n_head
        self.d_head = self.d_feat // self.n_head
        self.actv = actv
        self.use_bias = use_bias
        self.features = features
        self.rate = rate
        
        self.MHA = MultiHeadedAttention(self.d_feat, self.n_head, self.actv, self.use_bias)
        self.FFN = nn.Sequential([
            nn.Linear(self.d_feat, self.features, self.use_bias), 
            nn.ReUL(replace=True),
            nn.Linear(self.features, self.d_feat, self.use_bias)
        ])
        
        self.layernorm1 = nn.LayerNorm(self.d_feat)
        self.layernorm2 = nn.LayerNorm(self.d_feat)

        self.dropout1 = nn.Dropout(self.rate)
        self.dropout2 = nn.Dropout(self.rate)

    def forward(self, x, mask):
        out1, _ = self.MHA(x, x, x, mask)
        out1 = self.dropout1(out1)
        out1 = self.layernorm1(out1 + x)

        out2 = self.FFN(out1)
        out2 = self.dropout2(out2)
        out2 = self.layernorm2(out2 + out1)

        return out2

class DecoderLayer(nn.Module):
    def __init__(self, d_feat=128, n_head=5, actv=F.relu, use_bias=True, features=256, rate=0.1):
        super(EncoderLyaer, self).__init__()
        self.d_feat = d_feat
        self.n_head = n_head
        self.d_head = self.d_feat // self.n_head
        self.actv = actv
        self.use_bias = use_bias
        self.features = features
        self.rate = rate
        
        self.MHA1 = MultiHeadedAttention(self.d_feat, self.n_head, self.actv, self.use_bias)
        self.MHA2 = MultiHeadedAttention(self.d_feat, self.n_head, self.actv, self.use_bias)
        self.FFN = nn.Sequential([
            nn.Linear(self.d_feat, self.features, self.use_bias), 
            nn.ReUL(replace=True),
            nn.Linear(self.features, self.d_feat, self.use_bias)
        ])
        
        self.layernorm1 = nn.LayerNorm(self.d_feat)
        self.layernorm2 = nn.LayerNorm(self.d_feat)
        self.layernorm3 = nn.LayerNorm(self.d_feat)

        self.dropout1 = nn.Dropout(self.rate)
        self.dropout2 = nn.Dropout(self.rate)
        self.dropout3 = nn.Dropout(self.rate)

    def forward(self, x, encoder_output, look_mask, padding_mask):
        out1, attn1 = self.MHA1(x, x, x, look_mask)
        out1 = self.dropout1(out1)
        out1 = self.layernorm1(out1 + x)

        out2, attn2 = self.MHA2(encoder_output, encoder_output, out1, padding_mask)
        out2 = self.dropout2(out2)
        out2 = self.layernorm2(out2 + out1)

        out3 = self.FFN(out2)
        out3 = self.dropout3(out3)
        out3 = self.layernorm3(out3 + out2)

        return out3, attn1, attn2