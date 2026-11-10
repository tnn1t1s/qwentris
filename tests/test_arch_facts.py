"""SPEC §1 architecture facts, checked against the pinned transformers.

Covers the scalar facts and the code structure. The exact-overwrite-after-1e6-writes
test needs the compiled mem_write primitive and lives in test_primitives.py (P1).
"""

import inspect

import torch
import torch.nn.functional as F
import transformers
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

from qwentris.compiler.graph import A_LOG, L  # the values the compiler writes into the weights


def tiny_config(**kw):
    base = dict(
        vocab_size=16,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        layer_types=["linear_attention"] * 2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    base.update(kw)
    return Qwen3_5TextConfig(**base)


def test_pinned_version():
    assert transformers.__version__ == "5.22.1"


def test_all_linear_attention_model_builds():
    m = Qwen3_5ForCausalLM(tiny_config())
    assert all(type(layer.linear_attn).__name__ == "Qwen3_5GatedDeltaNet" for layer in m.model.layers)
    assert m.config.hidden_act == "silu"


def test_no_biases_in_mixer_or_mlp():
    m = Qwen3_5ForCausalLM(tiny_config())
    names = [n for n, _ in m.named_parameters()]
    assert not any(n.endswith(".bias") for n in names)


def test_rmsnorm_is_one_plus_w_with_zero_init():
    n = mq.Qwen3_5RMSNorm(4)
    assert torch.all(n.weight == 0)
    x = torch.tensor([[1.0, -2.0, 3.0, 0.5]])
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.equal(n(x), ref)


def test_beta_rounds_to_one_and_zero():
    assert torch.sigmoid(torch.tensor(L)).item() == 1.0
    assert torch.sigmoid(torch.tensor(-L)).item() < 1e-12


def test_decay_is_exactly_one():
    a_log = torch.tensor(A_LOG)
    dt_bias = torch.tensor(1.0)  # observation: transformers initialises dt_bias to ones
    for a in (0.0, L, -L):
        g = -a_log.exp() * F.softplus(torch.tensor(a) + dt_bias)
        assert g.exp().item() == 1.0


def test_silu_drive_levels():
    assert F.silu(torch.tensor(L)).item() == L
    assert F.silu(torch.tensor(0.0)).item() == 0.0


def test_l2norm_of_scaled_onehot_is_exact_and_zero_stays_zero():
    x = torch.zeros(1, 1, 1, 32)
    x[..., 7] = L
    y = mq.l2norm(x)
    assert y[..., 7].item() == 1.0 and y.abs().sum().item() == 1.0
    assert mq.l2norm(torch.zeros(1, 1, 1, 32)).abs().sum().item() == 0.0


def test_recurrent_path_is_read_after_write_delta_rule():
    src = inspect.getsource(mq.torch_recurrent_gated_delta_rule)
    assert "kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)" in src
    assert "delta = (v_t - kv_mem) * beta_t" in src
    i_update = src.index("last_recurrent_state = last_recurrent_state + k_t")
    i_read = src.index("core_attn_out[:, :, i] = (last_recurrent_state * q_t")
    assert i_update < i_read
    assert "query = query / (query.shape[-1] ** 0.5)" in src


def test_single_token_decode_takes_recurrent_path():
    src = inspect.getsource(mq.Qwen3_5GatedDeltaNet.forward)
    assert "if recurrent_state is not None and seq_len == 1:" in src
    assert "repeat_interleave(self.num_v_heads // self.num_k_heads" in src


def test_conv_has_no_bias_and_silu_follows():
    m = Qwen3_5ForCausalLM(tiny_config())
    attn = m.model.layers[0].linear_attn
    assert attn.conv1d.bias is None and attn.conv1d.kernel_size == (4,)
    assert attn.activation == "silu"
