# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion import pipeline_stable_diffusion
import torch

# Adapted
from diffusers.models.attention_processor import Attention, XFormersAttnProcessor, SpatialNorm
from diffusers.utils.import_utils import is_xformers_available
from typing import Callable, Optional
from diffusers.utils import deprecate

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class MPAdapterAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        processor: Optional["MPAdapterProcessor"] = None,
        out_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(
                num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(
                f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            self.to_k = nn.Linear(self.cross_attention_dim,
                                  self.inner_kv_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim,
                                  self.inner_kv_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(
                added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(
                added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(
                    added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(
                nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(
                self.inner_dim, self.out_dim, bias=out_bias)

        self.processor = MPAdapterProcessor()


class MPAdapterProcessor(nn.Module):
    def __init__(self,
                 added_kv_injt_num: Optional[int] = 0,
                 only_text_attention: Optional[bool] = False,
                 attention_op: Optional[Callable] = None,
                 query_dim: torch.Tensor = None,
                 cross_attention_dim: torch.Tensor = None,
                 device: Optional[str] = None,
                 dtype: Optional[str] = None,
                 ):

        super().__init__()
        self.attention_op = attention_op
        self.added_kv_injt_num = added_kv_injt_num
        self.only_text_attention = only_text_attention
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim
        self.pixel_embeds = None

        # In accordance with the augmentation of target concept number
        # the injection of additional KVProjection layers becomes requisite
        if not only_text_attention and added_kv_injt_num > 0:
            self.to_mp_k = nn.ModuleList([nn.Linear(
                self.cross_attention_dim or self.query_dim, self.query_dim, bias=False) for _ in range(added_kv_injt_num)]).to(device=device, dtype=dtype)
            self.to_mp_v = nn.ModuleList([nn.Linear(
                self.cross_attention_dim or self.query_dim, self.query_dim, bias=False) for _ in range(added_kv_injt_num)]).to(device=device, dtype=dtype)

    def __call__(
        self,
        attn: MPAdapterAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        pixel_embeds: Optional[list] = None,
        added_pixel_prompt: Optional[int] = -1,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(
            attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )

        # Adapted
        if not self.only_text_attention and added_pixel_prompt != -1 and self.added_kv_injt_num > 0:
            mp_key = self.to_mp_k[added_pixel_prompt](pixel_embeds[added_pixel_prompt])
            mp_value = self.to_mp_v[added_pixel_prompt](pixel_embeds[added_pixel_prompt])
            mp_key = attn.head_to_batch_dim(mp_key).contiguous()
            mp_value = attn.head_to_batch_dim(mp_value).contiguous()
            
            mp_hidden_states = xformers.ops.memory_efficient_attention(
                    query, mp_key, mp_value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
                )
        
            hidden_states = 0.5 * hidden_states + 0.8 * mp_hidden_states
        
        #
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(
                -1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class _XFormersAttnProcessor(XFormersAttnProcessor):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        pixel_embeds: Optional[list] = None,
        added_pixel_prompt: Optional[int] = 0,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return super().__call__(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )
