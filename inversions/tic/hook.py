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
        
        
class AttentionQKVInverter:
    def __init__(self, 
                 unet, 
                 cross_attn: Optional[list] = None, 
                 self_attn: Optional[list] = None,
                 cross_start_idx: int = 0,
                 self_start_idx: int = 0,
                 invert_q: bool = True,
                 invert_k: bool = True,
                 invert_v: bool = True):
        self.unet = unet
        self.cross_attn = cross_attn
        self.self_attn = self_attn
        self.cross_start_idx = cross_start_idx
        self.self_start_idx = self_start_idx
        self.invert_q = invert_q
        self.invert_k = invert_k
        self.invert_v = invert_v
        
        self.cross_cursor = 0
        self.self_cursor = 0
        
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
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

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
            # --------- custom code: invert q, k, v ---------
            # -----------------------------------------------
            if attn.is_cross_attention:
                if self.cross_attn is not None:
                    if self.cross_cursor >= self.cross_start_idx:
                        query_, key_, value_ = self.cross_attn[0]
                    else:
                        query_, key_, value_ = None, None, None
                    del self.cross_attn[0]
                else:
                    query_, key_, value_ = None, None, None
                self.cross_cursor += 1
            else:
                if self.self_attn is not None:
                    if self.self_cursor >= self.self_start_idx:
                        query_, key_, value_ = self.self_attn[0]
                    else:
                        query_, key_, value_ = None, None, None
                    del self.self_attn[0]
                else:
                    query_, key_, value_ = None, None, None
                self.self_cursor += 1
            
            if self.invert_q and query_ is not None:
                query = query_
            if self.invert_k and key_ is not None:
                key = key_
            if self.invert_v and value_ is not None:
                value = value_
            # -----------------------------------------------
            # --------- custom code: invert q, k, v ---------
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
