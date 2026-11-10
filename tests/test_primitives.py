"""P1 acceptance: each primitive compiled alone into a stock Qwen3_5ForCausalLM,
run token by token with a cache, matches the NumPy reference after binarisation
with margin >= 0.4, and keeps the residual norm constant across tokens."""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

from qwentris.compiler import primitives as P
from qwentris.compiler.emit_hf import Runner, build, load
from qwentris.compiler.graph import G, Program
from qwentris.compiler.layout import bits, not_, relabel
from qwentris.ref.tetris import SHAPES, X_RANGE

MIN_MARGIN = 0.4  # invariant: SPEC §4 P1
NORM_RTOL = 1e-5  # invariant: SPEC §3.2
# parameter: writes in the exact-overwrite test; SPEC §4 P1 asks for 10^6
N_OVERWRITES = int(os.environ.get("QWENTRIS_OVERWRITES", "1000000"))


def run_checked(prog, tokens):
    """Compile, run, compare. Returns (runner, min margin, emitted tokens)."""
    config, sd = build(prog)
    runner = Runner(load(config, sd))
    refs = prog.reference(tokens)
    computed = [d for f in prog.fields if f.stage > 0 for d in f.dims]
    index = {t: i for i, t in enumerate(prog.vocab)}
    margin, norms, emitted = 0.5, [], []
    for tok, ref in zip(tokens, refs):
        logits, layers = runner.step(index[tok])
        err = np.abs(layers[-1][computed] - ref[computed]).max() if computed else 0.0
        margin = min(margin, 0.5 - err)
        norms.append(np.linalg.norm(layers, axis=1))
        if prog.emit_fn is not None:
            got = prog.vocab[int(np.argmax(logits))]
            assert got == prog.reference_next(ref), (tok, got)
            emitted.append(got)
    norms = np.array(norms)
    rel = np.abs(norms / norms[0] - 1).max()
    assert rel < NORM_RTOL, f"residual norm varies by {rel:.2e}"
    assert margin >= MIN_MARGIN, f"margin {margin:.3f}"
    return runner, margin, emitted


def test_fp32_unit_identities():
    g = torch.tensor(G)
    assert (F.silu(g) / g).item() == 1.0
    assert abs(F.silu(-g).item()) < 1e-11
    assert F.silu(torch.tensor(0.0)).item() == 0.0


def test_threshold_and_or_not_two_levels():
    prog = Program()
    x = prog.input_dualrail("x", 6)
    a3 = P.and_(prog, "a3", [P.lit(x, 0), P.lit(x, 1, False), P.lit(x, 2)])
    o4 = P.or_(prog, "o4", [P.lit(x, 1), P.lit(x, 3), P.lit(not_(x), 4), P.lit(x, 5)])
    maj = P.threshold(prog, "maj", [P.lit(x, b) for b in range(5)], 3)
    two = P.thresholds(prog, "two", [([P.lit(x, 0)], 1), ([P.lit(x, 1), P.lit(x, 2)], 2)])
    # level 2: consumes level-1 outputs, including a NOT view
    deep = P.and_(prog, "deep", [P.lit(not_(a3)), P.lit(o4), P.lit(bits(two, [1]))])
    prog.set_vocab(range(64), lambda t: {x: t})
    runner, margin, _ = run_checked(prog, list(range(64)))
    assert runner.model.config.num_hidden_layers == 2 and deep.stage == 4 and maj.stage == 2
    # spot-check semantics against Python, not just against the reference functions
    refs = prog.reference(range(64))
    for t in range(64):
        b = [t >> i & 1 for i in range(6)]
        assert a3.decode(refs[t]) == (b[0] and not b[1] and b[2])
        assert o4.decode(refs[t]) == (b[1] or b[3] or not b[4] or b[5])
        assert maj.decode(refs[t]) == (sum(b[:5]) >= 3)


def test_lut_piece_shapes_and_linear_maps():
    prog = Program()
    typ = prog.input_onehot("type", tuple("IOTSZJL"))
    rot = prog.input_onehot("rot", range(4))
    x = prog.input_onehot("x", tuple(X_RANGE))

    def bitmap(p, r):
        return sum(1 << (4 * rr + cc) for rr, cc in SHAPES[p][r])

    shape = P.lut(prog, "shape", [typ, rot], bitmap, "dualrail", range(16))
    xr = P.linear_map(prog, "x+1", x, lambda u: u + 1 if u + 1 in X_RANGE else "OOB", tuple(X_RANGE) + ("OOB",))
    rot_next = relabel(rot, lambda r: (r + 1) % 4)  # the dim for rot r now reads as rot+1 mod 4
    shape_cw = P.lut(prog, "shape_cw", [typ, rot_next], bitmap, "dualrail", range(16))
    toks = [(p, r, xx) for p in "IOTSZJL" for r in range(4) for xx in X_RANGE]
    prog.set_vocab(toks, lambda t: {typ: t[0], rot: t[1], x: t[2]})
    run_checked(prog, toks)
    refs = prog.reference(toks)
    for (p, r, xx), v in zip(toks, refs):
        assert shape.decode(v) == bitmap(p, r)
        assert shape_cw.decode(v) == bitmap(p, (r + 1) % 4)
        assert xr.decode(v) == (xx + 1 if xx < 9 else "OOB")


def test_mux_fields_and_constants():
    prog = Program()
    sel = prog.input_onehot("sel", range(6))
    ins = [prog.input_dualrail(f"in{i}", 10) for i in range(5)]
    choices = {i: ins[i] for i in range(5)}
    choices[5] = 0b1010101010  # constant input
    out = P.mux(prog, "out", sel, choices, like=ins[0])
    rng = random.Random(0)
    toks = [(rng.randrange(6),) + tuple(rng.randrange(1024) for _ in range(5)) for _ in range(300)]
    prog.set_vocab(toks, lambda t: {sel: t[0], **{ins[i]: t[1 + i] for i in range(5)}})
    run_checked(prog, toks)
    for t, v in zip(toks, prog.reference(toks)):
        assert out.decode(v) == (t[1 + t[0]] if t[0] < 5 else 0b1010101010)


def build_memory_program():
    """20 single-slot row heads, a 20-slot display head (never read), a onehot register,
    and an MLP mux over the row reads."""
    prog = Program()
    rows = 20
    row_sel = [prog.input_onehot(f"wr{r}", (0,), optional=True) for r in range(rows)]
    row_any = prog.input_onehot("wrow", range(rows), optional=True)
    value = prog.input_dualrail("value", 10, optional=True)
    reg_w = prog.input_onehot("regw", (0,), optional=True)
    reg_v = prog.input_onehot("regv", tuple("IOTSZJL"), optional=True)
    pick = prog.input_onehot("pick", range(rows))
    reads = [P.memory(prog, f"B{r}", 1, row_sel[r], value) for r in range(rows)]
    P.memory(prog, "DISPLAY", rows, row_any, value, read=False)
    reg = P.memory(prog, "REG", 1, reg_w, reg_v)
    picked = P.mux(prog, "picked", pick, dict(enumerate(reads)), like=value)

    def embed(t):
        kind = t[0]
        a = {pick: t[-1]}
        if kind == "BOOT":
            a.update({f: 0 for f in row_sel})
            a.update({value: 0, reg_w: 0, reg_v: "I"})
        elif kind == "W":
            a.update({row_sel[t[1]]: 0, row_any: t[1], value: t[2]})
        elif kind == "R":
            a.update({reg_w: 0, reg_v: t[1]})
        return a

    return prog, embed, reads, reg, picked


def memory_tokens(n, seed):
    rng = random.Random(seed)
    toks = [("BOOT", 0)]
    for _ in range(n):
        k = rng.random()
        if k < 0.6:
            toks.append(("W", rng.randrange(20), rng.randrange(1024), rng.randrange(20)))
        elif k < 0.8:
            toks.append(("R", rng.choice("IOTSZJL"), rng.randrange(20)))
        else:
            toks.append(("N", rng.randrange(20)))
    return toks


def test_memory_read_after_write_and_display_state():
    prog, embed, reads, reg, picked = build_memory_program()
    # display head must hold the full board, so write every row before decoding it
    toks = [("BOOT", 0)] + [("W", r, 0, 0) for r in range(20)] + memory_tokens(400, 1)[1:]
    prog.set_vocab(sorted(set(toks), key=str), embed)
    runner, margin, _ = run_checked(prog, toks)
    refs = prog.reference(toks)
    # read values follow the writes
    board, regv = [0] * 20, "I"
    for t, v in zip(toks, refs):
        if t[0] == "W":
            board[t[1]] = t[2]
        if t[0] == "R":
            regv = t[1]
        assert [f.decode(v) for f in reads] == board
        assert reg.decode(v) == regv and picked.decode(v) == board[t[-1]]
    # the display head's recurrent state is the board: slot r row = L * dualrail(row r)
    disp = [h.name for h in prog.heads].index("DISPLAY")
    S = runner.state(0)[0, disp].numpy()  # [k_dim, v_dim]
    value_field = prog.heads[disp].write_values[0]
    decoded = [sum(1 << b for b in range(10) if S[r, 2 * b] > S[r, 2 * b + 1]) for r in range(20)]
    assert decoded == board
    print(f"\nmemory: {len(toks)} tokens, margin {margin:.6f}, {value_field.nbits}-bit rows")


def test_compiled_beta_and_decay_are_exact():
    prog, embed, *_ = build_memory_program()
    toks = memory_tokens(50, 2)
    prog.set_vocab(sorted(set(toks), key=str), embed)
    config, sd = build(prog)
    model = load(config, sd)
    attn = model.model.layers[0].linear_attn
    norm = model.model.layers[0].input_layernorm
    emb = model.model.embed_tokens.weight
    with torch.no_grad():
        x = norm(emb)
        beta = attn.in_proj_b(x).sigmoid()
        g = -attn.A_log.float().exp() * F.softplus(attn.in_proj_a(x).float() + attn.dt_bias)
    assert torch.all(g.exp() == 1.0)
    assert torch.all((beta == 1.0) | (beta < 1e-12))
    assert (beta == 1.0).any()


def test_exact_overwrite_after_many_writes():
    """Drive the compiled layer-0 q/k/v/beta/g through the HF recurrent kernel for
    N_OVERWRITES alternating writes to one slot, then check the slot holds the last value."""
    prog, embed, *_ = build_memory_program()
    w_a, w_b = ("W", 3, 0b1100110011, 0), ("W", 3, 0b0011001100, 0)
    prog.set_vocab([("BOOT", 0), w_a, w_b], embed)
    config, sd = build(prog)
    model = load(config, sd)
    layer = model.model.layers[0]
    attn = layer.linear_attn
    with torch.no_grad():
        x = layer.input_layernorm(model.model.embed_tokens.weight)  # [3, D]
        qkv = F.silu(attn.in_proj_qkv(x))  # conv weight [0,0,0,1] passes the current token
        H, kd, vd = attn.num_v_heads, attn.head_k_dim, attn.head_v_dim
        q, k, v = torch.split(qkv, [H * kd, H * kd, H * vd], dim=-1)
        q, k, v = q.view(3, H, kd), k.view(3, H, kd), v.view(3, H, vd)
        beta = attn.in_proj_b(x).sigmoid()
        g = -attn.A_log.exp() * F.softplus(attn.in_proj_a(x) + attn.dt_bias)
        state = None
        chunk = 20000  # parameter: tokens per kernel call; bounds memory, not results
        idx = torch.tensor([0] + [1, 2] * (N_OVERWRITES // 2))
        for s in range(0, len(idx), chunk):
            ii = idx[s : s + chunk]
            _, state = mq.torch_recurrent_gated_delta_rule(
                q[ii][None], k[ii][None], v[ii][None], g=g[ii][None], beta=beta[ii][None],
                initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
    head = 3  # B3
    last = v[2, head]
    assert torch.allclose(state[0, head, 0], last, rtol=0, atol=1e-5 * float(last.abs().max()))
    other = 4  # B4 was written only by BOOT
    assert torch.allclose(state[0, other, 0], v[0, other], rtol=0, atol=1e-5 * float(v[0, other].abs().max()))


def test_emit_structured_tokens():
    prog = Program()
    r_in = prog.input_onehot("r", range(20))
    b_in = prog.input_dualrail("b", 10)
    # computed fields that name the token to emit: row r+1 mod 20, bits inverted
    r_out = P.linear_map(prog, "r_out", r_in, lambda r: (r + 1) % 20, tuple(range(20)))
    one_sel = prog.input_onehot("one_sel", (0,))
    b_out = P.mux(prog, "b_out", one_sel, {0: not_(b_in)}, like=b_in)
    inputs = [(r, b) for r in range(20) for b in range(0, 1024, 37)]
    outs = [("WROW", r, b) for r in range(20) for b in range(1024)]
    prog.set_vocab(
        inputs + outs,
        lambda t: {r_in: t[0], b_in: t[1], one_sel: 0} if len(t) == 2 else {r_in: 0, b_in: 0, one_sel: 0},
    )
    prog.set_emit(lambda t: [(r_out, t[1]), (b_out, t[2])] if len(t) == 3 else None)
    _, _, emitted = run_checked(prog, inputs)
    assert emitted == [("WROW", (r + 1) % 20, b ^ 1023) for r, b in inputs]


def test_save_and_load_with_auto_class(tmp_path):
    prog, embed, *_ = build_memory_program()
    toks = memory_tokens(20, 3)
    prog.set_vocab(sorted(set(toks), key=str), embed)
    config, sd = build(prog)
    load(config, sd).save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    assert type(loaded).__name__ == "Qwen3_5ForCausalLM"
    got = loaded.state_dict()
    for k, v in sd.items():
        assert torch.equal(got[k], v), k
