# %%
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# Image(B, 3, H, W) 
# -> Patch (B, P, 3, H_P, W_P) -> (B, P, 3*H_P*W_P)
# -> Linear Projection (MLP) (B, P, D) 
# -> Patch Embedding + Position Embedding + cls embedding 
# -> Transformer Encoder x L -> MLP -> Classification
# %%
class ScaledDotProductAttention(nn.Modulesd):
    # (B, D, D) -> (B, D, D_v)
    # q, k, v: 3 dim
    # q shape == k shape (B, D1, D_k)
    # v shape (B, D1, D_v)

    def forward(self, query, key, value):
        dk = key.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / np.sqrt(dk)
        attention = F.softmax(scores, dim=-1)
        out = attention.matmul(value)
        return out, attention

y = torch.rand(1, 28, 28)
out = ScaledDotProductAttention()(y, y, torch.rand(1, 28, 28))
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
    
    def forward(self, q, k, v):
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
        scaled_attention, attention_weights = ScaledDotProductAttention()(q, k, v)
        # print(scaled_attention.shape, attention_weights.shape)

        scaled_attention = scaled_attention.permute([0, 2, 1, 3])

        concat_attention = scaled_attention.reshape(batch_size, -1, self.features)

        out = self.fc(concat_attention)
        if self.act is not None:
            out = self.act(out)

        return out, attention_weights # (batch_size, seq_len_q, features), (batch_size, num_head, seq_len_q, seq_len_k)

temp_mha = MultiHeadAttention(features=28, num_heads=4)
out, attn = temp_mha(q=y, k=y, v=torch.rand(1, 28, 28))
print(out.shape, attn.shape)

# %%
class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, mlp_dim, out_features=None, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        actual_out_dim = in_features if out_features is None else out_features
        layer_list = [
            nn.Linear(in_features, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, actual_out_dim),
            nn.Dropout(dropout_rate)
        ]

        self.net = nn.Sequential(*layer_list)
    def forward(self, x):
        return self.net(x)

sample_ffn = FeedForwardNetwork(512, 2048)
sample_ffn(torch.rand(64, 50, 512)).shape

# %%
class Encoder1DBlock(nn.Module):
    def __init__(self, in_features, num_heads, mlp_dim, dropout_rate):
        super(Encoder1DBlock, self).__init__()
        self.ln1 = nn.LayerNorm(in_features)

        self.attn = MultiHeadAttention(in_features, num_heads)

        self.dropout = nn.Dropout(dropout_rate)

        self.ln2 = nn.LayerNorm(in_features)
        self.ffn = FeedForwardNetwork(in_features, mlp_dim, dropout_rate=dropout_rate)

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
    def __init__(self, in_feature, num_layers, num_heads, mlp_dim, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.encoder  = nn.Sequential(*[Encoder1DBlock(in_feature, num_heads, mlp_dim, dropout_rate) for i in range(num_layers)])

        self.ln = nn.LayerNorm(in_feature)

    def forward(self, x):
        out = self.encoder(x)
        out = self.ln(out)
        return out

sample_En = Encoder(784, 3, 7, 1024)
sample_En(torch.rand(16, 5, 784)).shape

# %%
class ViT(nn.Module):
    def __init__(self, img_size, img_channels, patch_size, in_feature, num_classes, num_layers, num_heads, mlp_dim, dropout_rate=0.1):
        super(ViT, self).__init__()

        assert img_size % patch_size == 0, 'Image size must be divisible by the patch size.'
        num_patches = (img_size // patch_size) ** 2
        patch_dim = img_channels * patch_size ** 2
        assert num_patches > 16, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.img_size = img_size
        self.patch_size = patch_size

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, in_feature))
        self.patch_to_embed = nn.Linear(patch_dim, in_feature)
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_feature))
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer_encoder = Encoder(in_feature, num_layers, num_heads, mlp_dim, dropout_rate)

        self.to_cls_token = nn.Identity()

        layers = []
        layers.append(nn.LayerNorm(in_feature))
        layers.append(nn.Linear(in_feature, mlp_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(mlp_dim, num_classes))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, img):
        
        slices = self.img_size // self.patch_size
        # image to patch
        out = []
        batch_size = img.shape[0]
        for i in range(slices):
            for j in range(slices):
                out.append(img[:,:,i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size].reshape(batch_size, -1))
        out = torch.stack(out, dim=1)
        
        out = self.patch_to_embed(out)
        
        b, n, _ = out.shape

        out += self.pos_embed[:, :n]
        out = self.dropout(out)

        out = self.transformer_encoder(out)

        out = self.to_cls_token(out[:, 0])
        return self.mlp_head(out)

# %%
sample_ViT = ViT(256, 3, 16, 128, 10, 6, 8, 16) 
sample_ViT(torch.randn(1, 3, 256, 256)).shape