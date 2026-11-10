"""P2 acceptance: the compiled Qwentris model plays the protocol exactly.

The full 10 000-game differential test runs on a GPU with python -m qwentris.difftest.
These tests run the same checks on a CPU at a size that
fits one core.
"""

import os

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM

from qwentris.build import compile_qwentris
from qwentris.compiler.emit_hf import Runner, build, load
from qwentris.difftest import run_batch
from qwentris.games import play
from qwentris.ref import isa

# parameter: reference-evaluator games (pure NumPy, ~0.5 ms per token)
N_REF_RANDOM = int(os.environ.get("QWENTRIS_P2_REF_RANDOM", "50"))
N_REF_GREEDY = int(os.environ.get("QWENTRIS_P2_REF_GREEDY", "5"))
# parameter: compiled-model games on CPU. The stock depthwise conv1d runs as a per-channel
# loop on CPU (observed ~150 ms per token on the laptop), so laptop runs stay short.
N_HF_PIECES = int(os.environ.get("QWENTRIS_P2_HF_PIECES", "3"))


@pytest.fixture(scope="module")
def prog():
    return compile_qwentris()


@pytest.fixture(scope="module")
def model(prog, tmp_path_factory):
    """Compiled, saved with save_pretrained, reloaded with AutoModelForCausalLM."""
    config, sd = build(prog)
    path = tmp_path_factory.mktemp("qwentris")
    load(config, sd).save_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    assert type(m).__name__ == "Qwen3_5ForCausalLM"
    for k, v in m.state_dict().items():
        assert torch.equal(v, sd[k]), k
    return m


def check_reference(prog, trace):
    step = prog.reference_stepper()
    for i, tok in enumerate(trace):
        x = step(tok)
        if tok not in isa.STOP_TOKENS and i + 1 < len(trace):
            assert prog.decode_next(x) == trace[i + 1], (i, tok)


def test_reference_circuit_matches_isa(prog):
    lines = 0
    for seed in range(N_REF_RANDOM):
        check_reference(prog, play(seed, "random")[0])
    for seed in range(N_REF_GREEDY):
        trace = play(seed, "greedy", max_pieces=100)[0]
        check_reference(prog, trace)
        lines += sum(t[1] for t in trace if t[0] == "LINES")
    print(f"\nreference: {N_REF_RANDOM} random + {N_REF_GREEDY} greedy games, {lines} lines")


def test_emit_is_unique(prog):
    """On ideal residuals, exactly one vocabulary token's emit pattern matches."""
    step = prog.reference_stepper()
    trace = play(3, "greedy", max_pieces=8)[0]
    seen = set()
    for i, tok in enumerate(trace):
        x = step(tok)
        if tok in isa.STOP_TOKENS or i + 1 >= len(trace) or trace[i + 1][0] in seen:
            continue
        seen.add(trace[i + 1][0])
        assert prog.reference_next(x) == trace[i + 1]
    assert seen == {"SETP", "PHASE", "WROW", "TICK_END", "NEED_PIECE", "SRC", "LINES"}


def test_compiled_model_residuals_and_display(prog, model):
    """Single-sequence recurrent path: residual equals the reference, argmax equals isa,
    norm constant, DISPLAY state decodes to the board after every token."""
    index = {t: i for i, t in enumerate(prog.vocab)}
    trace = play(0, "random", max_pieces=N_HF_PIECES)[0]
    step, run = prog.reference_stepper(), Runner(model)
    fill = prog.fill.dims[0]
    board, worst = [0] * 20, 0.0
    for i, tok in enumerate(trace):
        x = step(tok)
        logits, layers = run.step(index[tok])
        err = np.abs(layers[-1] - x)
        err[fill] = 0.0
        worst = max(worst, float(err.max()))
        assert worst < 0.1, (i, tok)
        for li, h in enumerate(layers):
            norm2 = prog.norm2_at(2 * li + 2)
            assert abs(float(h @ h) - norm2) / norm2 < 1e-5, (i, li)
        if tok[0] == "WROW":
            board[tok[1]] = tok[2]
        S = run.state(0)[0, prog.display_head].numpy()
        assert [sum(1 << b for b in range(10) if S[r, 2 * b] > S[r, 2 * b + 1]) for r in range(20)] == board
        if tok not in isa.STOP_TOKENS and i + 1 < len(trace):
            assert prog.vocab[int(np.argmax(logits))] == trace[i + 1], (i, tok)
    print(f"\ncompiled: {len(trace)} tokens, max residual error {worst:.2e}")


def test_batched_difftest_detects_a_wrong_trace(prog, model):
    games = [("r0", play(0, "random", max_pieces=2)[0])]
    bad = list(games[0][1])
    i = next(i for i, t in enumerate(bad) if t[0] == "SETP" and i > 5)
    bad[i] = (*bad[i][:5], 1 - bad[i][5])
    games.append(("corrupt", bad))
    ok, wrong = run_batch(model, prog, games)
    assert ok["first_token_mismatch"] == -1 and ok["first_board_mismatch"] == -1
    assert ok["min_logit_gap"] > 29.0
    assert wrong["first_token_mismatch"] == i - 1
