"""Turn a Program into a stock Qwen3_5ForCausalLM config and state dict, and run it.

Layer 0's mixer holds every memory head; every other mixer is a no-op
(out_proj = 0). RMSNorm weights stay 0, so each norm multiplies by (1 + 0) and
the compiler folds the known constant RMSNorm scale into the next projection.
"""

import math

import numpy as np
import torch
from transformers import DynamicCache, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from qwentris.compiler.graph import A_LOG, L, LOGIT_SCALE, RMS_EPS

CONV_KERNEL = 4  # invariant: Qwen3.5 linear_conv_kernel_dim; weight [0,0,0,1] passes the current token


def build(prog):
    """-> (Qwen3_5TextConfig, state_dict of float32 tensors)."""
    emb = prog.embed_matrix()  # also allocates FILL and fixes the embedding norm
    D, V, n_layers = prog.ndims, len(prog.vocab), prog.n_layers
    H = max(1, len(prog.heads))
    k_dim = max([h.n_slots for h in prog.heads] + [1])
    v_dim = max([len(h.value_dims) for h in prog.heads] + [1])
    inter = max([len(prog.units[l]) for l in range(n_layers)] + [1])

    config = Qwen3_5TextConfig(
        vocab_size=V,
        hidden_size=D,
        intermediate_size=inter,
        num_hidden_layers=n_layers,
        layer_types=["linear_attention"] * n_layers,
        linear_num_key_heads=H,
        linear_num_value_heads=H,
        linear_key_head_dim=k_dim,
        linear_value_head_dim=v_dim,
        linear_conv_kernel_dim=CONV_KERNEL,
        rms_norm_eps=RMS_EPS,
        hidden_act="silu",
        tie_word_embeddings=False,
        pad_token_id=None,
    )
    sd = {"model.embed_tokens.weight": emb}
    one = prog.one.dims[0]
    conv_dim = 2 * H * k_dim + H * v_dim

    for i in range(n_layers):
        p = f"model.layers.{i}."
        qkv = np.zeros((conv_dim, D))
        z = np.zeros((H * v_dim, D))
        b = np.zeros((H, D))
        out = np.zeros((D, H * v_dim))
        if i == 0:
            s = prog.rms_scale(0)
            for h_i, h in enumerate(prog.heads):
                q0, k0, v0 = h_i * k_dim, H * k_dim + h_i * k_dim, 2 * H * k_dim + h_i * v_dim
                qkv[q0 + h.read_slot, one] = L / s
                for j, d in enumerate(h.write_slot.dims):
                    qkv[k0 + j, d] = L / s
                    b[h_i, d] = 2 * L / s
                b[h_i, one] = -L / s
                for j, d in enumerate(h.value_dims):
                    qkv[v0 + j, d] = L / s
                z[h_i * v_dim : (h_i + 1) * v_dim, one] = L / s
                if h.outs is not None:
                    # read: o = L*val/sqrt(k_dim); RMSNormGated divides by sqrt(mean(o^2)+eps), times silu(z)=L
                    o = L / math.sqrt(k_dim)
                    factor = o / math.sqrt(o * o * h.value_count / v_dim + RMS_EPS) * L
                    for j, d in enumerate(h.out_dims):
                        out[d, h_i * v_dim + j] = 1.0 / factor
        conv = np.zeros((conv_dim, 1, CONV_KERNEL))
        conv[:, 0, -1] = 1.0
        sd[p + "linear_attn.in_proj_qkv.weight"] = qkv
        sd[p + "linear_attn.in_proj_z.weight"] = z
        sd[p + "linear_attn.in_proj_b.weight"] = b
        sd[p + "linear_attn.in_proj_a.weight"] = np.zeros((H, D))
        sd[p + "linear_attn.conv1d.weight"] = conv
        sd[p + "linear_attn.A_log"] = np.full(H, A_LOG)
        sd[p + "linear_attn.dt_bias"] = np.zeros(H)
        sd[p + "linear_attn.norm.weight"] = np.ones(v_dim)
        sd[p + "linear_attn.out_proj.weight"] = out

        s = prog.rms_scale(2 * i + 1)
        gate, up, down = np.zeros((inter, D)), np.zeros((inter, D)), np.zeros((D, inter))
        for u_i, u in enumerate(prog.units[i]):
            for d, w in u.gate.items():
                gate[u_i, d] = w / s
            for d, w in u.up.items():
                up[u_i, d] = w / s
            for d, w in u.down.items():
                down[d, u_i] = w
        sd[p + "mlp.gate_proj.weight"] = gate
        sd[p + "mlp.up_proj.weight"] = up
        sd[p + "mlp.down_proj.weight"] = down
        sd[p + "input_layernorm.weight"] = np.zeros(D)
        sd[p + "post_attention_layernorm.weight"] = np.zeros(D)

    sd["model.norm.weight"] = np.zeros(D)
    s = prog.rms_scale(2 * n_layers)
    head = np.zeros((V, D))
    for t_i, tok in enumerate(prog.vocab):
        comps = prog.emit_fn(tok) if prog.emit_fn else None
        if comps is None:
            head[t_i, one] = -LOGIT_SCALE / s
            continue
        n = 0
        for f, v in comps:
            for d in f.encode(v):
                head[t_i, d] += LOGIT_SCALE / s
                n += 1
        head[t_i, one] -= LOGIT_SCALE * n / s
    sd["lm_head.weight"] = head
    return config, {k: torch.tensor(v, dtype=torch.float32) for k, v in sd.items()}


def load(config, state_dict):
    model = Qwen3_5ForCausalLM(config)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


class Runner:
    """Token-by-token driver over the stock HF forward with a cache (the recurrent path)."""

    def __init__(self, model):
        self.model = model
        self.cache = DynamicCache(config=model.config)
        self.pos = 0
        self.layer_out = []
        for layer in model.model.layers:
            layer.register_forward_hook(lambda m, i, o: self.layer_out.append(o[0] if isinstance(o, tuple) else o))

    @torch.no_grad()
    def step(self, tok_id):
        """-> (logits [V], residual after each layer [n_layers, D]) as numpy."""
        self.layer_out = []
        ids = torch.tensor([[tok_id]])
        # observation (transformers 5.22.1): with only linear_attention layers and a cache,
        # the model raises in cache.get_seq_length() unless position_ids is given and the
        # full-attention mask is skipped. DeltaNet ignores positions, and the recurrent mask
        # for an unpadded batch is None, so both are passed explicitly.
        pos = torch.tensor([[self.pos]])
        self.pos += 1
        masks = {"linear_attention": None}
        out = self.model(
            input_ids=ids, position_ids=pos, attention_mask=masks, past_key_values=self.cache, use_cache=True
        )
        self.cache = out.past_key_values
        layers = torch.stack([h[0, -1] for h in self.layer_out]).numpy()
        return out.logits[0, -1].numpy(), layers

    def state(self, layer=0):
        return self.cache.get_recurrent_state(layer, state_idx=0)


class BatchRunner:
    """B independent sequences stepped together through the stock HF forward with one cache.

    Returns per-step argmax ids, the top-1 minus top-2 logit gap, and the recurrent state of
    one head in layer 0, all as tensors on the model's device.
    """

    def __init__(self, model, batch):
        self.model = model
        self.batch = batch
        self.cache = DynamicCache(config=model.config)
        self.pos = 0
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def step(self, tok_ids, head):
        ids = torch.as_tensor(tok_ids, device=self.device).view(self.batch, 1)
        pos = torch.full((self.batch, 1), self.pos, device=self.device)
        self.pos += 1
        out = self.model(input_ids=ids, position_ids=pos, attention_mask={"linear_attention": None},
                         past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        self.cache = out.past_key_values
        top = out.logits[:, -1].topk(2, dim=-1)
        state = self.cache.get_recurrent_state(0, state_idx=0)[:, head]
        return top.indices[:, 0], top.values[:, 0] - top.values[:, 1], state
