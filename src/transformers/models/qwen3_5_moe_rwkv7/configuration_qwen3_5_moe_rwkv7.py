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
from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring
from ..qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig


@auto_docstring
@strict
class Qwen3_5MoeRwkv7Config(Qwen3_5MoeTextConfig):
    r"""
    Configuration for a Qwen3.5-MoE language model whose token mixers are replaced by RWKV-7.

    `linear_attention` layers use `rwkv_linear_head_dim` while `full_attention` layers use
    `rwkv_full_head_dim`. Both kinds of layers share `rwkv_attention_hidden_size`, which allows the
    per-token RWKV-7 `v_first` value to flow through every decoder layer.

    Args:
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Retained for Qwen3.5-MoE checkpoint metadata compatibility; it is unused after GDN replacement.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Original Qwen GDN key head size, retained for configuration compatibility.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Original Qwen GDN value head size used when inferring the shared RWKV-7 projection width.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Original Qwen GDN key head count, retained for configuration compatibility.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Original Qwen GDN value head count used when inferring the shared RWKV-7 projection width.
        rwkv_attention_hidden_size (`int`, *optional*):
            Width of the RWKV-7 projections. By default this is inferred from the original Qwen3.5-MoE
            GDN and GQA projection widths, which must agree.
        rwkv_linear_head_dim (`int`, *optional*, defaults to 128):
            RWKV-7 head size used for layers marked as `linear_attention`.
        rwkv_full_head_dim (`int`, *optional*, defaults to 256):
            RWKV-7 head size used for layers marked as `full_attention`.
        rwkv_decay_low_rank_dim (`int`, *optional*, defaults to 64):
            Low-rank width of the RWKV-7 decay projection.
        rwkv_gate_low_rank_dim (`int`, *optional*, defaults to 128):
            Low-rank width of the RWKV-7 output gate projection.
        rwkv_a_low_rank_dim (`int`, *optional*, defaults to 64):
            Low-rank width of the RWKV-7 in-context learning-rate projection.
        rwkv_v_low_rank_dim (`int`, *optional*, defaults to 32):
            Low-rank width of the RWKV-7 value-residual projection.
        rwkv_layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon used by the standard LayerNorm modules around token mixing, MoE, and the final output.
    """

    model_type = "qwen3_5_moe_rwkv7"

    base_model_tp_plan = {
        "layers.*.attn.r_proj": "colwise",
        "layers.*.attn.k_proj": "colwise",
        "layers.*.attn.v_proj": "colwise",
        "layers.*.attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    rwkv_attention_hidden_size: int | None = None
    rwkv_linear_head_dim: int = 128
    rwkv_full_head_dim: int = 256
    rwkv_decay_low_rank_dim: int = 64
    rwkv_gate_low_rank_dim: int = 128
    rwkv_a_low_rank_dim: int = 64
    rwkv_v_low_rank_dim: int = 32
    rwkv_layer_norm_eps: float = 1e-5

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        supported_layer_types = {"linear_attention", "full_attention"}
        unsupported_layer_types = set(self.layer_types) - supported_layer_types
        if unsupported_layer_types:
            raise ValueError(
                "Qwen3_5MoeRwkv7Config only supports Qwen layer types "
                f"{sorted(supported_layer_types)}, got {sorted(unsupported_layer_types)}."
            )

        original_linear_width = self.linear_num_value_heads * self.linear_value_head_dim
        original_full_width = self.num_attention_heads * self.head_dim
        if self.rwkv_attention_hidden_size is None:
            if original_linear_width != original_full_width:
                raise ValueError(
                    "Cannot infer a shared RWKV-7 attention width because the original Qwen GDN and GQA widths "
                    f"differ ({original_linear_width} and {original_full_width}). Set "
                    "`rwkv_attention_hidden_size` explicitly."
                )
            self.rwkv_attention_hidden_size = original_linear_width

        if self.rwkv_attention_hidden_size <= 0:
            raise ValueError("`rwkv_attention_hidden_size` must be positive.")
        for name in ("rwkv_linear_head_dim", "rwkv_full_head_dim"):
            head_dim = getattr(self, name)
            if head_dim <= 0 or self.rwkv_attention_hidden_size % head_dim != 0:
                raise ValueError(
                    f"`rwkv_attention_hidden_size` ({self.rwkv_attention_hidden_size}) must be divisible by "
                    f"`{name}` ({head_dim})."
                )


__all__ = ["Qwen3_5MoeRwkv7Config"]
