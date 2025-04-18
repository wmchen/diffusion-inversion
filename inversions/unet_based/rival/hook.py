import torch
import torch.nn.functional as F
from typing import Optional

from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import Attention


class AttentionQKVHook:
    def __init__(self, unet: UNet2DConditionModel):
        assert isinstance(unet, UNet2DConditionModel)
        self.unet = unet
        self.cross_attn = []
        self.self_attn = []
        
    def apply(self):
        def parse(net, block_type):
            if net.__class__.__name__ == "Attention":
                net.forward = self.custom_forward(net)
            elif hasattr(net, "children"):
                for net in net.children():
                    parse(net, block_type)
        
        for name, module in self.unet.named_children():
            if "down" in name:
                parse(module, "down")
            elif "mid" in name:
                parse(module, "mid")
            elif "up" in name:
                parse(module, "up")
    
    def remove(self):
        def parse(net, block_type):
            if net.__class__.__name__ == "Attention":
                net.forward = self.origin_forward(net)
            elif hasattr(net, "children"):
                for net in net.children():
                    parse(net, block_type)
        
        for name, module in self.unet.named_children():
            if "down" in name:
                parse(module, "down")
            elif "mid" in name:
                parse(module, "mid")
            elif "up" in name:
                parse(module, "up")
    
    def origin_forward(self, attn: Attention):
        
        def forward(hidden_states: torch.Tensor, 
                    encoder_hidden_states: Optional[torch.Tensor] = None, 
                    attention_mask: Optional[torch.Tensor] = None,
                    temb: Optional[torch.Tensor] = None):
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
        
        return forward
        
    def custom_forward(self, attn: Attention):
        
        def forward(hidden_states: torch.Tensor, 
                    encoder_hidden_states: Optional[torch.Tensor] = None, 
                    attention_mask: Optional[torch.Tensor] = None,
                    temb: Optional[torch.Tensor] = None):
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
                
            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
                
            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
                
            query = attn.to_q(hidden_states)
            
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
            # -----------------------------------------------
            # ---------- custom code: save q, k, v ----------
            # -----------------------------------------------
            if attn.is_cross_attention:
                self.cross_attn.append((query, key, value))
            else:
                self.self_attn.append((query, key, value))
            # -----------------------------------------------
            # ---------- custom code: save q, k, v ----------
            # -----------------------------------------------
                
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
        
        return forward
        
    def clear_attn(self):
        self.cross_attn = []
        self.self_attn = []
        
        
class AttentionSelfAttnKVInjector(AttentionQKVHook):
    def __init__(self, unet: UNet2DConditionModel, t_align: int):
        self.unet = unet
        self.t_align = t_align
        self.self_attn = None
        self.cur_timestep = None

    def custom_forward(self, attn: Attention):
        
        def forward(hidden_states: torch.Tensor, 
                    encoder_hidden_states: Optional[torch.Tensor] = None, 
                    attention_mask: Optional[torch.Tensor] = None,
                    temb: Optional[torch.Tensor] = None):
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
                
            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
                
            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
                
            query = attn.to_q(hidden_states)
            
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            # -----------------------------------------------------------------------
            # ---------- custom code: Cross-Image Self-Attention Injection ----------
            # -----------------------------------------------------------------------
            if not attn.is_cross_attention:
                _, inv_key, inv_value = self.self_attn.pop(0)
                if self.cur_timestep > self.t_align:
                    # replace
                    key[1, :] = inv_key
                    value[1, :] = inv_value
                else:
                    # concatenation in the spatial dimension
                    key_cond = torch.cat([key[1, :].unsqueeze(0), inv_key], dim=1)
                    key_uncond = torch.cat([key[0, :].unsqueeze(0)] * 2, dim=1)
                    key = torch.cat([key_uncond, key_cond], dim=0)
                    value_cond = torch.cat([value[1, :].unsqueeze(0), inv_value], dim=1)
                    value_uncond = torch.cat([value[0, :].unsqueeze(0)] * 2, dim=1)
                    value = torch.cat([value_uncond, value_cond], dim=0)
            # -----------------------------------------------------------------------
            # ---------- custom code: Cross-Image Self-Attention Injection ----------
            # -----------------------------------------------------------------------
                
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
        
        return forward
