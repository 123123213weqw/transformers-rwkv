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
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    from transformers.models.qwen3_5_moe_rwkv7 import (
        Qwen3_5MoeRwkv7Cache,
        Qwen3_5MoeRwkv7Config,
        Qwen3_5MoeRwkv7ForCausalLM,
    )


def get_config():
    return Qwen3_5MoeRwkv7Config(
        vocab_size=97,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=256,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=["linear_attention", "full_attention"],
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts_per_tok=2,
        num_experts=4,
        rwkv_decay_low_rank_dim=8,
        rwkv_gate_low_rank_dim=8,
        rwkv_a_low_rank_dim=8,
        rwkv_v_low_rank_dim=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


@require_torch
class Qwen3_5MoeRwkv7ModelTest(unittest.TestCase):
    def test_auto_classes_and_mixed_head_sizes(self):
        config = get_config()

        self.assertIsInstance(AutoConfig.for_model(config.model_type), Qwen3_5MoeRwkv7Config)
        self.assertEqual(AutoModel.from_config(config).__class__.__name__, "Qwen3_5MoeRwkv7Model")
        model = AutoModelForCausalLM.from_config(config)
        self.assertIsInstance(model, Qwen3_5MoeRwkv7ForCausalLM)
        self.assertEqual([layer.attn.head_dim for layer in model.model.layers], [128, 256])
        self.assertEqual([layer.attn.num_heads for layer in model.model.layers], [2, 1])

    def test_recurrent_cache_matches_full_forward(self):
        torch.manual_seed(0)
        model = Qwen3_5MoeRwkv7ForCausalLM(get_config()).to(torch_device).eval()
        input_ids = torch.tensor([[3, 4, 5, 6]], device=torch_device)

        with torch.no_grad():
            full_logits = model(input_ids, use_cache=False).logits
            prefill_outputs = model(input_ids[:, :3], use_cache=True)
            decode_outputs = model(
                input_ids[:, 3:],
                past_key_values=prefill_outputs.past_key_values,
                use_cache=True,
            )

        cache = decode_outputs.past_key_values
        self.assertIsInstance(cache, Qwen3_5MoeRwkv7Cache)
        self.assertEqual(cache.get_seq_length(), input_ids.shape[1])
        self.assertEqual(tuple(cache.attention_states[0].shape), (1, 2, 128, 128))
        self.assertEqual(tuple(cache.attention_states[1].shape), (1, 1, 256, 256))
        torch.testing.assert_close(decode_outputs.logits[:, -1], full_logits[:, -1], rtol=1e-4, atol=1e-5)

    def test_left_padding_does_not_update_recurrent_state(self):
        torch.manual_seed(0)
        model = Qwen3_5MoeRwkv7ForCausalLM(get_config()).to(torch_device).eval()
        padded_ids = torch.tensor([[0, 0, 3, 4]], device=torch_device)
        attention_mask = torch.tensor([[0, 0, 1, 1]], device=torch_device)

        with torch.no_grad():
            padded_logits = model(padded_ids, attention_mask=attention_mask, use_cache=False).logits
            unpadded_logits = model(padded_ids[:, 2:], use_cache=False).logits

        torch.testing.assert_close(padded_logits[:, -2:], unpadded_logits, rtol=1e-4, atol=1e-5)

    def test_router_loss_backward_and_generation(self):
        torch.manual_seed(0)
        model = Qwen3_5MoeRwkv7ForCausalLM(get_config()).to(torch_device)
        input_ids = torch.tensor([[3, 4, 5]], device=torch_device)
        outputs = model(
            input_ids,
            labels=input_ids,
            use_cache=False,
            output_router_logits=True,
        )

        self.assertEqual(tuple(outputs.logits.shape), (1, 3, model.config.vocab_size))
        self.assertEqual(len(outputs.router_logits), model.config.num_hidden_layers)
        self.assertIsNotNone(outputs.aux_loss)
        outputs.loss.backward()
        self.assertIsNotNone(model.model.layers[0].attn.r_proj.weight.grad)

        model.eval()
        with torch.no_grad():
            generated = model.generate(input_ids[:, :2], max_new_tokens=2, do_sample=False)
        self.assertEqual(tuple(generated.shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
