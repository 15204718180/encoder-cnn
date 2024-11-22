import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F


class Residual(nn.Module):#残差连接
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
#在前向传播过程中，首先将输入张量x传递给存储在属性fn中的函数或子模块，得到一个输出张量。然后将这个输出张量与输入张量x进行相加，
# 这就是典型的残差连接。这种连接方式有助于梯度的顺利传播和避免梯度消失或梯度爆炸的问题。

class PreNorm(nn.Module):#预层归一化
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
#在前向传播过程中，首先将输入张量x传递给LayerNorm层进行层归一化操作，然后将归一化后的张量作为输入传递给存储在属性fn中的函数或子模块。
# 最终返回经过函数处理后的张量。这种预层归一化的方式可以帮助加速模型的收敛和提高性能。

class FeedForward(nn.Module):#前馈神经网络
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):#循环depth次，构建depth个Transformer层
            self.layers.append(nn.ModuleList([#每个Transformer层包含了一个预处理（PreNorm）、残差连接（Residual）和注意力机制（Attention）或前馈神经网络（FeedForward）
                PreNorm(dim, Residual(Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout))),
                PreNorm(dim, Residual(FeedForward(
                    dim, mlp_head, dropout=dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):#循环depth-2次，构建depth-2个卷积层，用于CAF模式中的跳跃连接。
            self.skipcat.append(
                nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))

    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3),
                                           last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask=mask)
                x = ff(x)
                nl += 1

        return x

#如果模式是ViT，则执行以下操作：
#循环遍历每个Transformer层，首先应用注意力机制，然后应用前馈神经网络。更新输入张量x，直到所有Transformer层处理完毕。

#如果模式是CAF，则执行以下操作：
#初始化一个列表last_output用于存储每个Transformer层的输出。
#循环遍历每个Transformer层，对于序号大于1的层，执行跳跃连接操作。更新输入张量x，直到所有Transformer层处理完毕。


class ViT(nn.Module):
    def __init__(self, image_size, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head=16, dropout=0., emb_dropout=0., mode='ViT', out_dims=64):
        super().__init__()

        patch_dim = image_size ** 2 * near_band

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))#位置编码的参数，用于表示序列中每个位置的信息
        self.patch_to_embedding = nn.Linear(patch_dim, dim)#将图像块转换为 Transformer 模型输入的线性层
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))#CLS token，用于表示整个序列的信息

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp1 = nn.LayerNorm(dim)
        self.mlp2 = nn.Linear(dim, out_dims)
        self.mlp3 = nn.Linear(out_dims, num_classes)
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, x, mask=None):

        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        # embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x = self.patch_to_embedding(x)  # [b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(
            self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        x = self.to_latent(x[:, 0])

        # MLP classification layer
        x = self.mlp1(x)
        output = self.mlp2(x)
        x = self.mlp3(output)
        return output, x
