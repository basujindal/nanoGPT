# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Gemma model implementation."""

import re
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List, Optional, Sequence, Tuple, Union

from gemma import gemma_config
from gemma import tokenizer


def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
                    dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out


class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), dtype=torch.int8),
                # requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(out_features))
        else:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features)),
                # requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.linear(x, weight)
        return output


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        if quant:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), dtype=torch.int8),
                # requires_grad=False,
            )
            self.weight_scaler = nn.Parameter(torch.Tensor(num_embeddings))
        else:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)),
                # requires_grad=False,
            )
        self.quant = quant

    def forward(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler.unsqueeze(-1)
        output = F.embedding(x, weight)
        return output


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output


class GemmaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant: bool,
    ):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, quant)
        self.up_proj = Linear(hidden_size, intermediate_size, quant)
        self.down_proj = Linear(intermediate_size, hidden_size, quant)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class GemmaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        quant: bool,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5

        self.qkv_proj = Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            quant=quant)
        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            quant=quant)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],
                               dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        value = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        key = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # # [batch_size, n_local_heads, input_len, max_seq_len]
        # scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        # scores = scores.masked_fill(mask[:, None,:,:] == 0, float('-inf'))
        # scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # # [batch_size, n_local_heads, input_len, head_dim]
        # output = torch.matmul(scores, v)

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None,:,:], dropout_p=0.0, scale = self.scaling)
        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(batch_size, input_len, -1))


        output = self.o_proj(output)
        return output


class GemmaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.self_attn = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            quant=config.quant,
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,

        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            mask=mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(GemmaDecoderLayer(config))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                mask=mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):

    def __init__(
        self,
        config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0

        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim
        vocab_size = config.vocab_size

        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder = Embedding(vocab_size, config.hidden_size, config.quant)
        self.model = GemmaModel(config)

        # Pre-compute rotary embedding table.
        rope_theta = getattr(config, 'rope_theta', 10000)
        freqs_cis = precompute_freqs_cis(head_dim,
                                         max_seq_len * 2,
                                         theta=rope_theta)
        
        self.fewqs_cis = freqs_cis
        self.register_buffer('freqs_cis', freqs_cis)

    def forward(
        self,
        input_token_ids: torch.Tensor,
        # input_positions: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # freqs_cis = self.freqs_cis.index_select(0, input_positions)
        freqs_cis = self.freqs_cis[:input_token_ids.shape[1]]

        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_token_ids)
        # Gemma normalizes the embedding by sqrt(hidden_size).
        hidden_states = hidden_states * (self.config.hidden_size**0.5)

        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            mask=mask,
        )
        embedder_weight = self.embedder.weight
        logits = torch.matmul(hidden_states, embedder_weight.t())
        return logits


    def load_weights(self, model_path: str, device = "cpu"):
        self.load_state_dict(
            torch.load(
                model_path, weights_only=False, map_location=device
            )['model_state_dict'],
            strict=False,
        )

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, break_at_eos=False, eos_token_id=None):

        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if break_at_eos and idx_next.item() == eos_token_id:
                print("breaking at eos")
                break

        return idx

        
