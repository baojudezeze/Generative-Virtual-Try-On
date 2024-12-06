# Adapted from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import math
import torch
import torch.nn as nn

class PerceiverResampler(nn.Module):
    def __init__(
            self,
            dim=1024,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=16,
            embedding_dim=1280,
            output_dim=1024,
            ff_mult=4,
    ):
        super().__init__()

        self.latents = nn.Parameter(
            torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                self.attn(dim=dim, dim_head=dim_head, heads=heads),
                self.ffn(dim, ff_mult),
            ]) for _ in range(depth)
        ])

    # attention
    class attn(nn.Module):
        def __init__(self, *, dim, dim_head=64, heads=8):
            super().__init__()
            self.scale = dim_head**-0.5
            self.dim_head = dim_head
            self.heads = heads
            inner_dim = dim_head * heads

            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
            self.to_out = nn.Linear(inner_dim, dim, bias=False)

        def forward(self, x, latents):
            x = self.norm1(x)
            latents = self.norm2(latents)
            b, l, _ = latents.shape

            q = self.to_q(latents)
            kv_input = torch.cat((x, latents), dim=-2)
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)

            batch, seq_len, _ = q.shape
            q = q.view(batch, seq_len, self.heads, -1).transpose(1, 2)
            batch, seq_len, _ = k.shape
            k = k.view(batch, seq_len, self.heads, -1).transpose(1, 2)
            v = v.view(batch, seq_len, self.heads, -1).transpose(1, 2)

            scale = 1 / math.sqrt(math.sqrt(self.dim_head))
            weight = (q * scale) @ (k * scale).transpose(-2, -1)
            weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
            out = weight @ v
            out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
            return self.to_out(out)

    # feedforward
    def ffn(self, dim, mult=4):
        inner_dim = int(dim * mult)
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, x):

        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
      
        for attn, ffn in self.blocks:
            latents = attn(x, latents) + latents
            latents = ffn(latents) + latents 
          
        latents = self.proj_out(latents)
        return self.norm_out(latents)
      

class FFNResampler(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.ffn(image_embeds)
        return clip_extra_context_tokens
