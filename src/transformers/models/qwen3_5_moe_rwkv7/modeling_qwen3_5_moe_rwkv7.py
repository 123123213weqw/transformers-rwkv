# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeExperts,
    Qwen3_5MoePreTrainedModel,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTopKRouter,
    load_balancing_loss_func,
)
from .configuration_qwen3_5_moe_rwkv7 import Qwen3_5MoeRwkv7Config


logger = logging.get_logger(__name__)

RWKV7_EXP_HALF = 0.606531


class Qwen3_5MoeRwkv7Cache(Cache):
    """Recurrent RWKV-7 states for every mixed-head decoder layer."""

    def __init__(self, num_hidden_layers: int):
        super().__init__(layers=[])
        self.num_hidden_layers = num_hidden_layers
        self.attention_states: list[torch.Tensor | None] = [None] * num_hidden_layers
        self.previous_hidden_states: list[torch.Tensor | None] = [None] * num_hidden_layers
        self._seen_tokens = 0

    def __len__(self):
        return self.num_hidden_layers

    def get_layer_state(self, layer_idx: int):
        return self.previous_hidden_states[layer_idx], self.attention_states[layer_idx]

    def update(
        self,
        previous_hidden_state: torch.Tensor,
        attention_state: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.previous_hidden_states[layer_idx] = previous_hidden_state
        self.attention_states[layer_idx] = attention_state
        return previous_hidden_state, attention_state

    def advance(self, num_tokens: int):
        self._seen_tokens += num_tokens

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def get_max_length(self, layer_idx: int | None = None) -> int:
        return -1

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        return self._seen_tokens + query_length, 0

    def reset(self):
        self.attention_states = [None] * self.num_hidden_layers
        self.previous_hidden_states = [None] * self.num_hidden_layers
        self._seen_tokens = 0

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for states in (self.previous_hidden_states, self.attention_states):
            for layer_idx, state in enumerate(states):
                if state is not None:
                    states[layer_idx] = state.index_select(0, beam_idx.to(state.device))

    def batch_repeat_interleave(self, repeats: int):
        for states in (self.previous_hidden_states, self.attention_states):
            for layer_idx, state in enumerate(states):
                if state is not None:
                    states[layer_idx] = state.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        for states in (self.previous_hidden_states, self.attention_states):
            for layer_idx, state in enumerate(states):
                if state is not None:
                    states[layer_idx] = state.index_select(0, indices.to(state.device))

    def crop(self, max_length: int):
        if max_length == 0:
            self.reset()
        elif max_length != self._seen_tokens:
            raise ValueError("RWKV-7 recurrent cache states cannot be cropped without stored state snapshots.")

    @property
    def batch_size(self) -> int:
        for states in (self.previous_hidden_states, self.attention_states):
            for state in states:
                if state is not None:
                    return state.shape[0]
        return -1

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_initialized(self) -> bool:
        return any(state is not None for state in self.attention_states)


class Qwen3_5MoeRwkv7LoRA(nn.Module):
    """Two-linear RWKV-7 low-rank projection with checkpoint-compatible key names."""

    def __init__(self, input_size: int, low_rank: int, output_size: int, bias: bool):
        super().__init__()
        self.lora = nn.Sequential(
            nn.Linear(input_size, low_rank, bias=False),
            nn.Identity(),
            nn.Linear(low_rank, output_size, bias=bias),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lora[2](self.lora[0](hidden_states))


class Qwen3_5MoeRwkv7Attention(nn.Module):
    """Correctness-first, pure PyTorch RWKV-7 token mixer."""

    def __init__(self, config: Qwen3_5MoeRwkv7Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.attention_hidden_size = config.rwkv_attention_hidden_size
        self.head_dim = (
            config.rwkv_linear_head_dim
            if config.layer_types[layer_idx] == "linear_attention"
            else config.rwkv_full_head_dim
        )
        self.num_heads = self.attention_hidden_size // self.head_dim

        for parameter_name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, parameter_name, nn.Parameter(torch.zeros(1, 1, self.hidden_size)))

        self.k_k = nn.Parameter(torch.ones(self.attention_hidden_size))
        self.k_a = nn.Parameter(torch.ones(self.attention_hidden_size))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(self.hidden_size, self.attention_hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.attention_hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.attention_hidden_size, bias=False)
        self.o_proj = nn.Linear(self.attention_hidden_size, self.hidden_size, bias=False)
        self.w_lora = Qwen3_5MoeRwkv7LoRA(
            self.hidden_size,
            config.rwkv_decay_low_rank_dim,
            self.attention_hidden_size,
            bias=True,
        )
        self.a_lora = Qwen3_5MoeRwkv7LoRA(
            self.hidden_size,
            config.rwkv_a_low_rank_dim,
            self.attention_hidden_size,
            bias=True,
        )
        self.g_lora = Qwen3_5MoeRwkv7LoRA(
            self.hidden_size,
            config.rwkv_gate_low_rank_dim,
            self.attention_hidden_size,
            bias=False,
        )
        if layer_idx != 0:
            self.v_lora = Qwen3_5MoeRwkv7LoRA(
                self.hidden_size,
                config.rwkv_v_low_rank_dim,
                self.attention_hidden_size,
                bias=True,
            )
        self.g_norm = nn.GroupNorm(
            self.num_heads,
            self.attention_hidden_size,
            eps=self.head_dim * 1e-5,
        )

    def _step(
        self,
        hidden_states: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        v_first: torch.Tensor,
        attention_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        difference = previous_hidden_state - hidden_states
        mixed = {
            name: hidden_states + difference * getattr(self, f"x_{name}").view(1, self.hidden_size)
            for name in ("r", "w", "k", "v", "a", "g")
        }

        receptance = self.r_proj(mixed["r"])
        decay = self.w_lora(torch.tanh(mixed["w"]))
        key = self.k_proj(mixed["k"])
        value = self.v_proj(mixed["v"])
        in_context_rate = torch.sigmoid(self.a_lora(mixed["a"]))
        gate = self.g_lora(torch.sigmoid(mixed["g"]))

        normalized_key = F.normalize(
            (key * self.k_k.view(1, self.attention_hidden_size)).view(batch_size, self.num_heads, self.head_dim),
            dim=-1,
            p=2,
        ).view(batch_size, self.attention_hidden_size)
        key = key * (1 + (in_context_rate - 1) * self.k_a.view(1, self.attention_hidden_size))

        if self.layer_idx == 0:
            v_first = value
        else:
            value = value + (v_first - value) * torch.sigmoid(self.v_lora(mixed["v"]))

        decay = torch.exp(-RWKV7_EXP_HALF * torch.sigmoid(decay.float()))
        value_key = value.view(batch_size, self.num_heads, self.head_dim, 1) @ key.view(
            batch_size, self.num_heads, 1, self.head_dim
        )
        learning_update = (-normalized_key).view(batch_size, self.num_heads, self.head_dim, 1) @ (
            normalized_key * in_context_rate
        ).view(batch_size, self.num_heads, 1, self.head_dim)

        attention_state = attention_state.float()
        attention_state = (
            attention_state * decay.view(batch_size, self.num_heads, 1, self.head_dim)
            + attention_state @ learning_update.float()
            + value_key.float()
        )
        output = attention_state.to(hidden_states.dtype) @ receptance.view(
            batch_size, self.num_heads, self.head_dim, 1
        )
        output = output.view(batch_size, self.attention_hidden_size)
        output = F.group_norm(
            output,
            num_groups=self.num_heads,
            weight=self.g_norm.weight,
            bias=self.g_norm.bias,
            eps=self.head_dim * 1e-5,
        )
        residual_scale = (
            receptance.view(batch_size, self.num_heads, self.head_dim)
            * key.view(batch_size, self.num_heads, self.head_dim)
            * self.r_k.view(1, self.num_heads, self.head_dim)
        ).sum(dim=-1, keepdim=True)
        output = output + (residual_scale * value.view(batch_size, self.num_heads, self.head_dim)).view(
            batch_size, self.attention_hidden_size
        )
        output = self.o_proj(output * gate)
        return output, hidden_states, attention_state, v_first

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3_5MoeRwkv7Cache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        previous_hidden_state = None
        attention_state = None
        if past_key_values is not None:
            previous_hidden_state, attention_state = past_key_values.get_layer_state(self.layer_idx)
        if previous_hidden_state is None:
            previous_hidden_state = hidden_states.new_zeros(batch_size, self.hidden_size)
        else:
            previous_hidden_state = previous_hidden_state.to(hidden_states.device)
        if attention_state is None:
            attention_state = torch.zeros(
                batch_size,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                dtype=torch.float32,
                device=hidden_states.device,
            )
        else:
            attention_state = attention_state.to(hidden_states.device)

        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("Qwen3.5-MoE-RWKV7 expects a 2D padding mask.")
            attention_mask = attention_mask[:, -sequence_length:].to(dtype=torch.bool, device=hidden_states.device)

        outputs = []
        layer_v_first = []
        for token_idx in range(sequence_length):
            token_hidden_states = hidden_states[:, token_idx]
            token_v_first = (
                hidden_states.new_zeros(batch_size, self.attention_hidden_size)
                if self.layer_idx == 0
                else v_first[:, token_idx]
            )
            old_previous_hidden_state = previous_hidden_state
            old_attention_state = attention_state
            output, new_previous_hidden_state, new_attention_state, token_v_first = self._step(
                token_hidden_states,
                previous_hidden_state,
                token_v_first,
                attention_state,
            )
            if attention_mask is not None:
                token_mask = attention_mask[:, token_idx]
                vector_mask = token_mask[:, None]
                matrix_mask = token_mask[:, None, None, None]
                output = torch.where(vector_mask, output, torch.zeros_like(output))
                new_previous_hidden_state = torch.where(
                    vector_mask, new_previous_hidden_state, old_previous_hidden_state
                )
                new_attention_state = torch.where(matrix_mask, new_attention_state, old_attention_state)
                token_v_first = torch.where(vector_mask, token_v_first, torch.zeros_like(token_v_first))
            previous_hidden_state = new_previous_hidden_state
            attention_state = new_attention_state
            outputs.append(output)
            if self.layer_idx == 0:
                layer_v_first.append(token_v_first)

        if use_cache and past_key_values is not None:
            past_key_values.update(previous_hidden_state, attention_state, self.layer_idx)

        if self.layer_idx == 0:
            v_first = torch.stack(layer_v_first, dim=1)
        return torch.stack(outputs, dim=1), v_first


class Qwen3_5MoeRwkv7DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5MoeRwkv7Config, layer_idx: int):
        super().__init__()
        self.attn = Qwen3_5MoeRwkv7Attention(config, layer_idx)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rwkv_layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rwkv_layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3_5MoeRwkv7Cache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, v_first = self.attn(
            hidden_states,
            v_first=v_first,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states
        return hidden_states, v_first


class Qwen3_5MoeRwkv7PreTrainedModel(Qwen3_5MoePreTrainedModel):
    config: Qwen3_5MoeRwkv7Config
    _no_split_modules = ["Qwen3_5MoeRwkv7DecoderLayer"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen3_5MoeTopKRouter, index=0),
        "hidden_states": OutputRecorder(Qwen3_5MoeRwkv7DecoderLayer, index=0),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3_5MoeRwkv7Attention):
            for parameter_name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "r_k"):
                init.zeros_(getattr(module, parameter_name))
            init.ones_(module.k_k)
            init.ones_(module.k_a)
        elif isinstance(module, Qwen3_5MoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


@auto_docstring
class Qwen3_5MoeRwkv7Model(Qwen3_5MoeRwkv7PreTrainedModel):
    def __init__(self, config: Qwen3_5MoeRwkv7Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5MoeRwkv7DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rwkv_layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = Qwen3_5MoeRwkv7Cache(self.config.num_hidden_layers)
        if past_key_values is not None and not isinstance(past_key_values, Qwen3_5MoeRwkv7Cache):
            raise TypeError(
                f"Qwen3_5MoeRwkv7Model requires `Qwen3_5MoeRwkv7Cache`; received {type(past_key_values).__name__}."
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Disabling the cache.")
            use_cache = False
            past_key_values = None

        hidden_states = inputs_embeds
        v_first = None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, v_first = decoder_layer(
                hidden_states,
                v_first=v_first,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        if use_cache:
            past_key_values.advance(inputs_embeds.shape[1])

        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@auto_docstring
class Qwen3_5MoeRwkv7ForCausalLM(Qwen3_5MoeRwkv7PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    config: Qwen3_5MoeRwkv7Config

    def __init__(self, config: Qwen3_5MoeRwkv7Config):
        super().__init__(config)
        self.model = Qwen3_5MoeRwkv7Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "Qwen3_5MoeRwkv7Cache",
    "Qwen3_5MoeRwkv7ForCausalLM",
    "Qwen3_5MoeRwkv7Model",
    "Qwen3_5MoeRwkv7PreTrainedModel",
]
