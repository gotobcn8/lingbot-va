# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from typing import Callable, ClassVar
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
    and_masks,
    or_masks
)
from functools import partial
try:
    from flash_attn_interface import flash_attn_func
except:
    from flash_attn import flash_attn_func

__all__ = ['WanTransformer3DModel']


def custom_sdpa(q, k, v):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2))
    return out.transpose(1, 2)

# MAX_TEXT_LEN = 128
MAX_TEXT_LEN = 512

class FlexAttnFunc(nn.Module):
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, dynamic=True, 
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    attention_mask: ClassVar[BlockMask] = None
    cross_attention_mask: ClassVar[BlockMask] = None
    

    def __init__(
        self, 
        is_cross=False,
    ) -> None:
        super().__init__()
        self.is_cross = is_cross
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)
        
        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = FlexAttnFunc.cross_attention_mask if self.is_cross else FlexAttnFunc.attention_mask

        x_out = FlexAttnFunc.flex_attn(q_varlen, k_varlen, v_varlen, block_mask=block_mask, kernel_options = {
                                                    "BLOCK_M": 64,
                                                    "BLOCK_N": 64,
                                                    "BLOCK_M1": 32,
                                                    "BLOCK_N1": 64,
                                                    "BLOCK_M2": 64,
                                                    "BLOCK_N2": 32,
                                                })

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(
        latent_shape, 
        action_shape, 
        padded_length, 
        chunk_size,
        window_size,
        patch_size,
        device,
    ):
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape

        # latent has been patched
        latent_seq_id = torch.arange(B)[:, None, None, None].\
            expand(-1, L_F // patch_size[0], L_H // patch_size[1], L_W // patch_size[2]).flatten()
        action_seq_id = torch.arange(B)[:, None, None, None].expand(-1, A_F, A_H, A_W).flatten()
        # Batch size
        # = batch[0] , seq_id = 0
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)

        # latent frame chunk: even
        # action frame chunk: odd
        # simulate: latent0, action0, latent1, action2,....
        latent_frame_id = torch.arange(L_F)[None, :, None, None].expand(B, -1, L_H // patch_size[1], L_W // patch_size[2])[None].flatten()
        action_frame_id = torch.arange(A_F)[None, :, None, None].expand(B, -1, A_H, A_W)[None].flatten()
        frame_ids = torch.cat([latent_frame_id // chunk_size * 2] * 2 + [action_frame_id // chunk_size * 2 + 1] * 2)

        # latent noisy (0), latent clean (1), action noisy (0), action clean (1)
        noise_ids = torch.cat(
            [
                torch.zeros_like(latent_frame_id),
                torch.ones_like(latent_frame_id),
                torch.zeros_like(action_frame_id),
                torch.ones_like(action_frame_id),
            ]
        )
        # pad to consistent length
        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        mask_mod = FlexAttnFunc._get_mask_mod(seq_ids.long().to(device), frame_ids.long().to(device), noise_ids.long().to(device), window_size)
        block_mask = FlexAttnFunc.compiled_create_block_mask(
                mask_mod, 1, 1, len(seq_ids), len(seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.attention_mask = block_mask

        text_seq_ids = torch.arange(B)[:, None].expand(-1, 512).flatten()
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(seq_ids.long().to(device), text_seq_ids.long().to(device))
        block_mask_cross = FlexAttnFunc.compiled_create_block_mask(
                mask_mod_cross, 1, 1, len(seq_ids), len(text_seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.cross_attention_mask = block_mask_cross

    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids, text_seq_ids):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == text_seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (text_seq_ids[kv_idx] >= 0)
        return seq_mask
    
    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
        # Same batch, batch[0] = batch[0]
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (seq_ids[kv_idx] >= 0)
        
        # past and current frames can be seen
        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] <= frame_ids[q_idx])
        
        # Past only
        def block_causal_mask_exclude_self(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] < frame_ids[q_idx])
        
        # current chunk only
        def block_self_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] == frame_ids[q_idx])
        
        # standard self-attn: clean_q -> clean {<= t}, this cooperate with block_causal_mask
        def clean2clean_mask(
                b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)
        
        # noise can see past clean tokens only
        def noise2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)

        # noisy -> noisy, can see current noisy token only
        def noise2noise_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)
        
        # set a window chunk
        def block_window_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor, window_size: int
        ):
            return ((frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size)

        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))
        mask = or_masks(*mask_list)
        mask = and_masks(mask, seq_mask)
        mask = and_masks(mask, partial(block_window_mask, window_size=window_size))
        return mask
       
class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim,
                                        flip_sin_to_cos=True,
                                        downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim,
                                               time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim,
                                                       dim,
                                                       act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        # time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.f_dim, 2)[:(self.f_dim // 2)].double() / self.f_dim))
        h_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.h_dim, 2)[:(self.h_dim // 2)].double() / self.h_dim))
        w_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.w_dim, 2)[:(self.w_dim // 2)].double() / self.w_dim))
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class WanAttention(torch.nn.Module):

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode='torch',
    ):
        super().__init__()
        if attn_mode == 'torch':
            self.attn_op = custom_sdpa
        elif attn_mode == 'flashattn':
            self.attn_op = flash_attn_func
        elif attn_mode == 'flex':
            self.attn_op = FlexAttnFunc(cross_attention_dim_head is not None)
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, only support torch and flashattn"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList([
            torch.nn.Linear(self.inner_dim, dim, bias=True),
            torch.nn.Dropout(dropout),
        ])
        self.norm_q = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.attn_caches = {} if cross_attention_dim_head is None else None

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        is_pred = cache['is_pred']
        cache['mask'][is_pred] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            'k':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'v':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'id':
            torch.full((total_tolen, ), -1, device=device),
            "mask":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
            "is_pred":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)

            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]

            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]['id']
        mask = self.attn_caches[cache_name]['mask']

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name, key, value, is_pred):
        cache = self.attn_caches[cache_name]

        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)

        new_id = self._next_cache_id(cache_name)

        cache['k'][:, slots] = key
        cache['v'][:, slots] = value
        cache['mask'][slots] = True
        cache['id'][slots] = new_id
        cache['is_pred'][slots] = is_pred
        return slots

    def restore_cache(self, cache_name, slots):
        self.attn_caches[cache_name]['mask'][slots] = False

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
        return_attn_probs=False,
        attn_slice=None,
        attn_slice_mask=None,
    ):
        kv_cache = self.attn_caches[
            cache_name] if (self.attn_caches is not None) and (cache_name in self.attn_caches) else None

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(x.shape[0], x.shape[1],
                                                x.shape[2], -1, 2))
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        slots = None
        if kv_cache is not None and kv_cache['k'] is not None:
            slots = self.update_cache(cache_name,
                                      key,
                                      value,
                                      is_pred=(update_cache == 1))
            key_pool = self.attn_caches[cache_name]['k']
            value_pool = self.attn_caches[cache_name]['v']
            mask = self.attn_caches[cache_name]['mask']
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]

        hidden_states = self.attn_op(query, key, value)

        attn_probs = None
        if return_attn_probs:
            if attn_slice is not None:
                q_start, q_length, k_start, k_length = attn_slice
                query_for_probs = query[:, q_start:q_start + q_length]
                key_for_probs = key[:, k_start:k_start + k_length]
            else:
                query_for_probs = query
                key_for_probs = key

            scale = key.shape[-1] ** -0.5
            attn_scores = torch.einsum(
                'bqhd,bkhd->bhqk',
                query_for_probs.float(),
                key_for_probs.float(),
            ) * scale
            if attn_slice_mask is not None:
                attn_mask = attn_slice_mask.to(device=attn_scores.device, dtype=torch.bool)
                attn_scores = attn_scores.masked_fill(
                    ~attn_mask[None, None],
                    torch.finfo(attn_scores.dtype).min,
                )
                empty_rows = ~attn_mask.any(dim=-1)
                if empty_rows.any():
                    attn_scores[:, :, empty_rows, :] = 0
            attn_probs = attn_scores.softmax(dim=-1)

        if update_cache == 0:
            if kv_cache is not None and kv_cache['k'] is not None:
                self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        if return_attn_probs:
            return hidden_states, attn_probs
        return hidden_states


class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
    ):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = FP32LayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim,
                               inner_dim=ffn_dim,
                               activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5)

        # Frames Downsample

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
        return_attn_probs=False,
        attn_slice=None,
        attn_slice_mask=None,
    ) -> torch.Tensor:
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
            rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)
        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1. + scale_msa) +
                              shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states,
                                 norm_hidden_states,
                                 norm_hidden_states,
                                 rotary_emb,
                                 update_cache=update_cache,
                                 cache_name=cache_name,
                                 return_attn_probs=return_attn_probs,
                                 attn_slice=attn_slice,
                                 attn_slice_mask=attn_slice_mask)
        attn_probs = None
        if return_attn_probs:
            attn_output, attn_probs = attn_output
        hidden_states = (hidden_states.float() +
                         attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(
            hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states,
                                 encoder_hidden_states,
                                 encoder_hidden_states,
                                 None,
                                 update_cache=0,
                                 cache_name=cache_name)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) *
                              (1. + c_scale_msa) +
                              c_shift_msa).type_as(hidden_states)

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = (hidden_states.float() +
                         ff_output.float() * c_gate_msa).type_as(hidden_states)
        if return_attn_probs:
            return hidden_states, attn_probs
        return hidden_states

class HiddenMLP(nn.Module):
    def __init__(self, dim, target_dim = None, hidden_dim=None):
        super().__init__()
        target_dim = target_dim or dim
        self.use_residual = target_dim == dim
        if hidden_dim is None:
            hidden_dim = target_dim if dim == 3072 else dim * 2

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )
        self.net.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(
                module.weight,
                a=0,
                mode='fan_out',
                nonlinearity='relu',
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: (B, Tokens, D)
        # residual structure：predict "change frrom future compared with current."
        pred = self.net(x)
        if self.use_residual:
            pred = x + pred
        return pred

class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    TODO
    """
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
                                        # "patch_embedding", 
                                        "patch_embedding_mlp",
                                        "condition_embedder", 
                                        'condition_embedder_action',
                                        "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", 
                             "scale_shift_table", 
                             "scale_shift_table_action",
                             "norm1", 
                             'action_norm1',
                             'text_norm1',
                             "norm2", 
                             'action_norm2',
                             'text_norm2',
                             "norm3",
                             'action_norm3',
                             'text_norm3'
                             ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(self,
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-06,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
        trace_dimension = 4096,
        target_dim = 4096,
        enable_trace = False,
    ):
        r"""
        TODO
        """
        super().__init__()
        self.num_layers = num_layers
        self.trace_dimension = trace_dimension
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size,
                                      rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            inner_dim)
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        self.blocks = nn.ModuleList([
            WanTransformerBlock(inner_dim,
                                ffn_dim,
                                num_attention_heads,
                                cross_attn_norm,
                                eps,
                                attn_mode=attn_mode) for _ in range(num_layers)
        ])

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim,
                                  out_channels * math.prod(patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        self.target_dim = target_dim

        # if enable_trace:
        #     self.init_trace_modules(trace_dimension=trace_dimension,
        #                             target_dim=target_dim)
        # else:
            # self.trace_hidden_mlp = nn.Linear(
            #     self.trace_dimension,
            #     self.inner_dim,
            #     # dtype=data_type,
            # )

    def init_trace_modules(self,
                           trace_dimension=None,
                           target_dim=None,
                           dtype=None,
                           device=None):
        trace_dimension = trace_dimension or self.trace_dimension
        target_dim = target_dim or self.target_dim

        reference_param = next(self.parameters(), None)
        if reference_param is not None:
            device = device or reference_param.device
            dtype = dtype or reference_param.dtype

        if hasattr(self, "future_hidden_mlp") and hasattr(self, "motion_hidden_mlp"):
            self.future_hidden_mlp = HiddenMLP(self.inner_dim, trace_dimension)
            self.motion_hidden_mlp = HiddenMLP(self.inner_dim, trace_dimension)
            if device is not None or dtype is not None:
                self.future_hidden_mlp.to(device=device, dtype=dtype)
                self.motion_hidden_mlp.to(device=device, dtype=dtype)
            self.register_to_config(enable_trace=True,
                                    trace_dimension=trace_dimension,
                                    target_dim=target_dim)
            return

        self.trace_dimension = trace_dimension
        self.target_dim = target_dim

        # def build_trace_proj():
        #     return nn.Sequential(
        #         nn.Linear(self.trace_dimension, self.inner_dim),
        #         nn.GELU(),
        #         nn.Linear(self.inner_dim, target_dim),
        #     )

        # self.trace_proj = build_trace_proj()
        # self.motion_trace_proj = build_trace_proj()
        # self.future_trace_proj = build_trace_proj()
        # self.motion_align_proj = nn.Sequential(
        #     nn.Linear(self.inner_dim, target_dim),
        #     nn.GELU(),
        #     nn.Linear(target_dim, target_dim),
        # )
        # self.future_hidden_mlp = nn.Linear(
        #     trace_dim,
        #     self.inner_dim,
        #     # dtype=data_type,
        # )
        # self.future_align_proj = nn.Sequential(
        #     nn.Linear(self.inner_dim, target_dim),
        #     nn.GELU(),
        #     nn.Linear(target_dim, target_dim),
        # )
        # self.trace_simple_proj = nn.Linear(self.trace_dimension,target_dim)
        self.future_hidden_mlp = HiddenMLP(self.inner_dim, trace_dimension)
        self.motion_hidden_mlp = HiddenMLP(self.inner_dim, trace_dimension)
        if device is not None or dtype is not None:
            self.future_hidden_mlp.to(device=device, dtype=dtype)
            self.motion_hidden_mlp.to(device=device, dtype=dtype)

        self.register_to_config(enable_trace=True,
                                trace_dimension=trace_dimension,
                                target_dim=target_dim)


    def _init_trace_parameters(self, 
            # trace_dim = 4096, 
            # target_dim = 2048, 
            K_frames = 1, 
            align_layer = 20,
            future_align_layer = 22,
            motion_align_layer = 15,
            teacher_temperature = 1.0,
            teacher_use_delta = True,
        ):
        # self.trace_hidden_mlp = nn.Linear(
        #     trace_dim,
        #     c,
        #     dtype=data_type,
        # )
        # self.trace_proj = nn.Sequential(
        #     nn.Linear(trace_dim, self.inner_dim, dtype=data_type,),
        #     nn.GELU(),
        #     nn.Linear(self.inner_dim, target_dim, dtype=data_type,)
        # )
        # self.align_repr_proj = nn.Sequential(
        #     nn.Linear(self.inner_dim, target_dim, dtype=data_type,),
        #     nn.GELU(),
        #     nn.Linear(target_dim, target_dim, dtype=data_type,),
        # )
        self.attn_teacher_temperature = teacher_temperature
        self.attn_teacher_use_delta = teacher_use_delta
        self.K_frames = K_frames
        self.align_layer = align_layer
        self.future_align_layer = future_align_layer
        self.motion_align_layer = motion_align_layer
        if self.align_layer < 1 and self.align_layer > 0:
            self.align_layer = int(self.align_layer * self.num_layers) - 1
        if self.future_align_layer < 1 and self.future_align_layer > 0:
            self.future_align_layer = int(self.future_align_layer * self.num_layers) - 1
        if self.motion_align_layer < 1 and self.motion_align_layer > 0:
            self.motion_align_layer = int(self.motion_align_layer * self.num_layers) - 1
        
        # print(self.align_layer)

    def clear_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(self, cache_name, attn_window,
                           latent_token_per_chunk, action_token_per_chunk,
                           device, dtype, batch_size):
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2) * action_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(cache_name, total_tolen,
                                      self.num_attention_heads,
                                      self.attention_head_dim, device, dtype, batch_size)
    
    def _input_embed(self, latents, input_type='latent'):
        # shape: (1, 48, 5, 24, 20)
        # patches (1, 2, 2)
        if input_type == 'latent':
            hidden_states = rearrange(
                latents,
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            hidden_states = self.patch_embedding_mlp(hidden_states)
        elif input_type == 'action':
            hidden_states = rearrange(latents, 'b c f h w -> b (f h w) c')
            hidden_states = self.action_embedder(hidden_states)
        elif input_type == 'text':
            hidden_states = self.condition_embedder.text_embedder(latents)
        # elif input_type == 'trace':
        #     if not hasattr(self, "trace_proj"):
        #         raise ValueError("Trace modules are not initialized.")
        #     hidden_states = self.trace_proj(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def downsample_trace(self, trace, target_frames):
        trace = rearrange(
            trace,
            'b (f p1) (h p2) (w p3) c -> b f (h w) (c p1 p2 p3)',
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2]
        )
        trace = trace.permute(0, 2, 3, 1)  # (B, Tokens, D, F)
        T = trace.shape[1]
        trace = rearrange(trace, 'b t d f -> (b t) d f')
        trace = F.interpolate(trace, size=target_frames, mode='linear', align_corners=False)
        trace = rearrange(trace, '(b t) d f -> b (f t) d', t = T)
        # trace = trace.permute(0, 3, 1, 2)  # (B, F_target, T, D)
        # trace = rerange(trace, 'b f t d -> b (f t) d')
        return trace, T

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (H // pach_scale_h) *
            (W // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    # future_alignment_loss_excludeself,
    # motion_incremental_alignment_tokenwise

    def _resolve_attention_align_layer(self, input_dict):
        layer = input_dict.get(
            'attn_align_layer',
            getattr(self, 'attn_align_layer', getattr(self, 'motion_align_layer', self.num_layers - 1)),
        )
        if isinstance(layer, float) and 0 < layer < 1:
            layer = int(layer * self.num_layers) - 1
        layer = int(layer)
        if layer < 0 or layer >= self.num_layers:
            raise ValueError(f"attn_align_layer must be in [0, {self.num_layers - 1}], got {layer}")
        return layer

    def _prepare_attention_teacher(self, input_dict, target_frames, device):
        if 'ta_teacher' in input_dict:
            ta_teacher = input_dict['ta_teacher'].to(device=device)
            if ta_teacher.dim() != 3:
                raise ValueError(f"Expected ta_teacher with shape [B, F, N], got {ta_teacher.shape}")
            return None, ta_teacher, ta_teacher.shape[1], ta_teacher.shape[2]

        trace_key = 'ta_feat' if 'ta_feat' in input_dict else 'trace'
        if trace_key not in input_dict:
            raise ValueError("forward_train_attn requires input_dict['trace'], input_dict['ta_feat'], or input_dict['ta_teacher']")

        ta_feat = input_dict[trace_key].to(device=device, dtype=torch.bfloat16).detach()
        if ta_feat.dim() == 5:
            ta_feat, tokens = self.downsample_trace(ta_feat, target_frames)
            ta_feat = rearrange(ta_feat, 'b (f n) d -> b f n d', f=target_frames, n=tokens)
        elif ta_feat.dim() == 4:
            tokens = ta_feat.shape[2]
        elif ta_feat.dim() == 3:
            tokens = input_dict.get('ta_tokens', None)
            if tokens is None:
                raise ValueError("ta_tokens is required when ta_feat has shape [B, F*N, D]")
            ta_feat = rearrange(ta_feat, 'b (f n) d -> b f n d', f=target_frames, n=tokens)
        else:
            raise ValueError(f"Expected trace/ta_feat with shape [B,F,H,W,C], [B,F,N,D], or [B,F*N,D], got {ta_feat.shape}")

        return ta_feat, None, ta_feat.shape[1], tokens

    def _latent_temporal_condition_mask(self, frames, chunk_size, window_size, device):
        frame_chunks = torch.arange(frames, device=device) // chunk_size * 2
        return (
            (frame_chunks[None, :] < frame_chunks[:, None])
            & ((frame_chunks[:, None] - frame_chunks[None, :]).abs() <= window_size)
        )

    def _noisy_to_condition_latent_mask(self, batch_size, frames, tokens, temporal_mask):
        batch_mask = torch.eye(batch_size, dtype=torch.bool, device=temporal_mask.device)
        token_mask = torch.ones(tokens, tokens, dtype=torch.bool, device=temporal_mask.device)
        mask = (
            batch_mask[:, None, None, :, None, None]
            & temporal_mask[None, :, None, None, :, None]
            & token_mask[None, None, :, None, None, :]
        )
        return mask.reshape(batch_size * frames * tokens, batch_size * frames * tokens)

    def _condition_key_attention_from_probs(
        self,
        attn_probs,
        latent_query_length,
        condition_key_length,
        batch_size,
        frames,
        tokens,
        temporal_mask,
    ):
        expected_latent_length = batch_size * frames * tokens
        if latent_query_length != expected_latent_length or condition_key_length != expected_latent_length:
            raise ValueError(
                "Latent attention slice does not match TA teacher shape: "
                f"query={latent_query_length}, key={condition_key_length}, "
                f"expected={expected_latent_length}"
            )

        if attn_probs.shape[0] != 1:
            raise ValueError(f"Expected flattened training batch in attention dim 0, got {attn_probs.shape[0]}")
        if attn_probs.shape[-2:] != (latent_query_length, condition_key_length):
            raise ValueError(
                "Expected local attention probs with shape [1, heads, noisy_len, condition_len], got "
                f"{attn_probs.shape}"
            )

        condition_probs = attn_probs[0].reshape(
            attn_probs.shape[1],
            batch_size,
            frames,
            tokens,
            batch_size,
            frames,
            tokens,
        )
        condition_probs = torch.stack(
            [condition_probs[:, batch_idx, :, :, batch_idx, :, :] for batch_idx in range(batch_size)],
            dim=1,
        )
        condition_probs = condition_probs * temporal_mask[None, None, :, None, :, None]
        return condition_probs.mean(dim=(0, 3))

    def _temporal_masked_attention_coverage_loss(
        self,
        condition_attention,
        temporal_mask,
        ta_feat=None,
        ta_teacher=None,
        teacher_temperature=1.0,
        teacher_use_delta=True,
        eps=1e-6,
    ):
        from .alignment import ta_to_attention_teacher

        if ta_teacher is None:
            if ta_feat is None:
                raise ValueError("Either ta_feat or ta_teacher must be provided")
            ta_teacher = ta_to_attention_teacher(ta_feat, use_delta=teacher_use_delta)
            ta_teacher = torch.softmax(ta_teacher / teacher_temperature, dim=-1)
        else:
            ta_teacher = ta_teacher.to(device=condition_attention.device)

        if condition_attention.shape[0] != ta_teacher.shape[0] or condition_attention.shape[-2:] != ta_teacher.shape[1:]:
            raise ValueError(
                "condition_attention and ta_teacher must agree on batch/key-frame/token dims, got "
                f"{condition_attention.shape} vs {ta_teacher.shape}"
            )

        teacher = ta_teacher[:, None] * temporal_mask[None, :, :, None]
        attention = condition_attention * temporal_mask[None, :, :, None]

        valid_queries = temporal_mask.any(dim=-1)
        if not valid_queries.any():
            return condition_attention.new_zeros(())

        attention = attention / (attention.sum(dim=(-2, -1), keepdim=True) + eps)
        teacher = teacher / (teacher.sum(dim=(-2, -1), keepdim=True) + eps)
        reward = (attention * teacher.detach()).sum(dim=(-2, -1))

        return -reward[:, valid_queries].mean()

    def forward_train_attn(self, input_dict, alignment_modules=None):
        input_dict['latent_dict']['noisy_latents'] = input_dict['latent_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['latent_dict']['latent'] = input_dict['latent_dict']['latent'].to(torch.bfloat16)
        input_dict['action_dict']['noisy_latents'] = input_dict['action_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['action_dict']['latent'] = input_dict['action_dict']['latent'].to(torch.bfloat16)

        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        batch_size = latent_dict['noisy_latents'].shape[0]
        latent_token_frames = latent_dict['noisy_latents'].shape[2] // self.patch_size[0]
        # attn_align_layer = self._resolve_attention_align_layer(input_dict)

        latent_hidden_states = self._input_embed(latent_dict['noisy_latents'], input_type='latent').flatten(0, 1)[None]
        action_hidden_states = self._input_embed(action_dict['noisy_latents'], input_type='action').flatten(0, 1)[None]
        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type='text').flatten(0, 1)[None]

        condition_latent_hidden_states = self._input_embed(latent_dict['latent'], input_type='latent').flatten(0, 1)[None]
        condition_action_hidden_states = self._input_embed(action_dict['latent'], input_type='action').flatten(0, 1)[None]

        ta_feat, ta_teacher, teacher_frames, teacher_tokens = self._prepare_attention_teacher(
            input_dict,
            latent_token_frames,
            latent_hidden_states.device,
        )

        hidden_states = torch.cat([
            latent_hidden_states,
            condition_latent_hidden_states,
            action_hidden_states,
            condition_action_hidden_states,
        ], dim=1)

        latent_grid_id = latent_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)
        rotary_emb = self.rope(full_grid_id)[:, :, None]

        latent_time_steps = torch.cat(
            [latent_dict['timesteps'].flatten(0, 1), latent_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        action_time_steps = torch.cat(
            [action_dict['timesteps'].flatten(0, 1), action_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        latent_temb, latent_timestep_proj = self._time_embed(
            latent_time_steps,
            latent_dict['noisy_latents'].shape[-2],
            latent_dict['noisy_latents'].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=False,
        )
        action_temb, action_timestep_proj = self._time_embed(
            action_time_steps,
            action_dict['noisy_latents'].shape[-2],
            action_dict['noisy_latents'].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=True,
        )
        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [
            latent_hidden_states.shape[1],
            condition_latent_hidden_states.shape[1],
            action_hidden_states.shape[1],
            condition_action_hidden_states.shape[1],
            padded_length,
        ]

        FlexAttnFunc.init_mask(
            latent_dict['noisy_latents'].shape,
            action_dict['noisy_latents'].shape,
            padded_length,
            input_dict["chunk_size"],
            window_size=input_dict['window_size'],
            patch_size=self.patch_size,
            device=hidden_states.device,
        )
        temporal_mask = self._latent_temporal_condition_mask(
            teacher_frames,
            input_dict["chunk_size"],
            input_dict['window_size'],
            hidden_states.device,
        )
        noisy_to_condition_mask = self._noisy_to_condition_latent_mask(
            batch_size,
            teacher_frames,
            teacher_tokens,
            temporal_mask,
        )

        loss_attn = None
        noisy_latent_len, condition_latent_len, _, _, _ = split_list
        noisy_latent_q_start = 0
        noisy_latent_q_length = noisy_latent_len
        condition_latent_k_start = noisy_latent_len
        condition_latent_k_length = condition_latent_len
        attn_slice = (
            noisy_latent_q_start,
            noisy_latent_q_length,
            condition_latent_k_start,
            condition_latent_k_length,
        )
        
        for layer, block in enumerate(self.blocks):
            capture_attn = layer == self.align_layer
            block_output = block(
                hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=False,
                return_attn_probs=capture_attn,
                attn_slice=attn_slice if capture_attn else None,
                attn_slice_mask=noisy_to_condition_mask if capture_attn else None,
            )
            if capture_attn:
                hidden_states, attn_probs = block_output
                condition_attention = self._condition_key_attention_from_probs(
                    attn_probs,
                    noisy_latent_q_length,
                    condition_latent_k_length,
                    batch_size,
                    teacher_frames,
                    teacher_tokens,
                    temporal_mask,
                )
                loss_attn = self._temporal_masked_attention_coverage_loss(
                    condition_attention,
                    temporal_mask,
                    ta_feat=ta_feat,
                    ta_teacher=ta_teacher,
                    teacher_temperature=self.attn_teacher_temperature,
                    teacher_use_delta=self.attn_teacher_use_delta,
                )
            else:
                hidden_states = block_output

        if loss_attn is None:
            raise ValueError(
                "Attention alignment layer was not reached. "
                f"attn_align_layer={attn_align_layer}, num_layers={self.num_layers}"
            )

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1. + scale) + shift).type_as(hidden_states)
        latent_hidden_states, _, action_hidden_states, _, _ = torch.split(hidden_states, split_list, dim=1)
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(
            latent_hidden_states,
            '1 (b l) (n c) -> b (l n) c',
            n=math.prod(self.patch_size),
            b=batch_size,
        )
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(action_hidden_states, '1 (b l) c -> b l c', b=batch_size)

        return latent_hidden_states, action_hidden_states, loss_attn
    
    def aligner(self, align_hidden_states, trace_hidden_states, tokens, align_modules, align_key = 'dynamic'):
        if align_modules is None or align_key not in align_modules:
            raise ValueError(f"Missing alignment module for key: {align_key}")

        # dynamic loss
        if align_key == 'dynamic':
            align_hidden_states = self.motion_hidden_mlp(align_hidden_states)
            align_loss = align_modules[align_key](
                align_hidden_states,
                trace_hidden_states,
                K=self.K_frames,
                Tokens=tokens,
            )
        
        elif align_key == 'future':
            align_hidden_states = self.future_hidden_mlp(align_hidden_states)
            align_loss = align_modules[align_key](
                align_hidden_states,
                trace_hidden_states,
                K=self.K_frames,
                Tokens=tokens,
            )
        else:
            raise ValueError(f"Unknown alignment key: {align_key}")

        return align_loss
        

    def forward_train_pre(self, input_dict, alignment_module = None):
        input_dict['latent_dict']['noisy_latents'] = input_dict['latent_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['latent_dict']['latent'] = input_dict['latent_dict']['latent'].to(torch.bfloat16)
        input_dict['action_dict']['noisy_latents'] = input_dict['action_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['action_dict']['latent'] = input_dict['action_dict']['latent'].to(torch.bfloat16)

        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        batch_size = latent_dict['noisy_latents'].shape[0]

        # noisy latents
        latent_hidden_states = self._input_embed(latent_dict['noisy_latents'], input_type='latent').flatten(0, 1)[None]
        action_hidden_states = self._input_embed(action_dict['noisy_latents'], input_type='action').flatten(0, 1)[None]
        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type='text')

        text_hidden_states = text_hidden_states.flatten(0, 1)[None]

        # conditional latent
        condition_latent_hidden_states = self._input_embed(latent_dict['latent'], input_type='latent').flatten(0, 1)[None]
        condition_action_hidden_states = self._input_embed(action_dict['latent'], input_type='action').flatten(0, 1)[None]
        
        # trace latents
        if 'trace' in input_dict:
            input_dict['trace'] = input_dict['trace'].to(torch.bfloat16)
            trace_hidden_states, tokens = self.downsample_trace(input_dict['trace'], latent_dict['noisy_latents'].shape[2])
            # 4096 -> 3072
            # trace_hidden_states = self._input_embed(trace_hidden_states, input_type='trace').flatten(0, 1)[None]

        hidden_states = torch.cat([latent_hidden_states, 
                                   condition_latent_hidden_states,
                                   action_hidden_states, 
                                   condition_action_hidden_states], dim=1)


        latent_grid_id = latent_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)

        rotary_emb = self.rope(full_grid_id)[:, :, None] 

        latent_time_steps = torch.cat(
            [latent_dict['timesteps'].flatten(0, 1), latent_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        action_time_steps = torch.cat(
            [action_dict['timesteps'].flatten(0, 1), action_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        latent_temb, latent_timestep_proj =self._time_embed(latent_time_steps, 
                        latent_dict['noisy_latents'].shape[-2], 
                        latent_dict['noisy_latents'].shape[-1], 
                        dtype=hidden_states.dtype, 
                        action_mode=False)
        action_temb, action_timestep_proj = self._time_embed(action_time_steps,
                        action_dict['noisy_latents'].shape[-2], 
                        action_dict['noisy_latents'].shape[-1], 
                        dtype=hidden_states.dtype, 
                        action_mode=True)
        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        # e.g. [360,360,264,264,32]
        split_list = [latent_hidden_states.shape[1], 
                      condition_latent_hidden_states.shape[1], 
                      action_hidden_states.shape[1], 
                      condition_action_hidden_states.shape[1],
                      padded_length]
        
        # noisy_latents: (1,48,3,24,20)
        # action_dict noisy_latents : (1, 30, 3, 88, 1)
        # padded_length = 32
        # chunk_size = 3
        # window_size = 9
        # patch_size = [1,2,2]
        FlexAttnFunc.init_mask(latent_dict['noisy_latents'].shape, 
                               action_dict['noisy_latents'].shape, 
                               padded_length, 
                               input_dict["chunk_size"],
                               window_size=input_dict['window_size'],
                               patch_size=self.patch_size,
                               device=hidden_states.device
                               )

        for layer, block in enumerate(self.blocks):
            # hidden_states = (1,1280,3072)
            # text_hidden_states = (1，33，3072) -> (1,128,3072)
            torch.cuda.synchronize()
            hidden_states = block(
                hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=False
            )
            torch.cuda.synchronize()
            if layer == self.align_layer:
                align_hidden_states, _, _, _, _ = torch.split(hidden_states, split_list, dim=1)

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(hidden_states)
        latent_hidden_states, _, action_hidden_states, _, _ = torch.split(hidden_states, split_list, dim=1)
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(
            latent_hidden_states,
            '1 (b l) (n c) -> b (l n) c',
            n=math.prod(self.patch_size), b=batch_size
        )  # batch, sequence = l, n = h * w, c = dim
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(action_hidden_states,
                                             '1 (b l) c -> b l c',
                                             b=batch_size)  #

        # cos_sim = F.cosine_similarity(align_hidden_states, h_new, dim=-1) / temperature
        # loss = 1 - cos_sim.mean()
        if 'trace' in input_dict:
            if not hasattr(self, "motion_hidden_mlp"):
                raise ValueError(
                    "Trace input was provided, but trace modules are not initialized. "
                    "Load the transformer with enable_trace=True or resume from a "
                    "checkpoint whose transformer/config.json has enable_trace=true."
                )
            align_hidden_states = self.motion_hidden_mlp(align_hidden_states)
            # trace_loss = alignment_module(align_hidden_states, trace_hidden_states, K = self.K_frames, Tokens = tokens)
            trace_loss = alignment_module(align_hidden_states, trace_hidden_states, K = self.K_frames, Tokens = tokens)
            return latent_hidden_states, action_hidden_states, trace_loss

        return latent_hidden_states, action_hidden_states

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
        train_attn_mode=False,
        alignment_module = None,
        alignment_modules = None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if train_attn_mode:
            return self.forward_train_attn(input_dict, alignment_modules or alignment_module)
        if train_mode:
            return self.forward_train_manifold(input_dict, alignment_modules or alignment_module)
        if action_mode:  # action input emb
            latent_hidden_states = rearrange(input_dict['noisy_latents'], 'b c f h w -> b (f h w) c')
            latent_hidden_states = self.action_embedder(
                latent_hidden_states
            )  # B L1 C
        else:  # latent input emb
            latent_hidden_states = rearrange(
                input_dict['noisy_latents'],
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            latent_hidden_states = self.patch_embedding_mlp(
                latent_hidden_states)
        text_hidden_states = self.condition_embedder.text_embedder(
            input_dict["text_emb"])  # B L2 C

        latent_grid_id = input_dict['grid_id']
        rotary_emb = self.rope(latent_grid_id)[:, :, None]  # 1 L 1 C
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])

        latent_time_steps = torch.repeat_interleave(
            input_dict['timesteps'],
            (input_dict['noisy_latents'].shape[-2] // pach_scale_h) *
            (input_dict['noisy_latents'].shape[-1] // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C

        for i, block in enumerate(self.blocks):
            latent_hidden_states = block(
                latent_hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name
            )

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (self.norm_out(latent_hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(latent_hidden_states)

        if action_mode:
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(latent_hidden_states,
                                             'b l (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size))  #

        return latent_hidden_states

    def set_requires_gradient_sync(self,should_asyc = False):
        if hasattr(self, "require_backward_grad_sync"):
            self.require_backward_grad_sync = should_asyc

# class LingbotTraceModel(nn.Module):
#     def __init__(
#         self,
#         transformer,
#     ):
#         super().__init__()
#         self.3Dtransformer = transformer

#     def forward(self):
#         pass

if __name__ == '__main__':
    model = WanTransformer3DModel(patch_size=[1, 2, 2],
                                  num_attention_heads=24,
                                  attention_head_dim=128,
                                  in_channels=48,
                                  out_channels=48,
                                  action_dim=30,
                                  text_dim=4096,
                                  freq_dim=256,
                                  ffn_dim=14336,
                                  num_layers=30,
                                  cross_attn_norm=True,
                                  eps=1e-6,
                                  rope_max_seq_len=1024,
                                  pos_embed_seq_len=None,
                                  attn_mode="torch")
    print(model)
