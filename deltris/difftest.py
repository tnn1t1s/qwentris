"""P2 differential test: the compiled model's trace equals the isa trace token for token,
and the board decoded from the DISPLAY head's recurrent state equals the oracle board
after every token.

The model is fed the isa trace (teacher forcing). Every token the model is asked to emit
is checked, so a full match is the same as the model producing the trace on its own.

    uv run python -m deltris.difftest --random 10000 --greedy 500 --out results/p2.jsonl
"""

import argparse
import json
import time

import numpy as np
import torch

from deltris.compiler.emit_hf import BatchRunner
from deltris.games import play
from deltris.ref import isa
from deltris.ref.tetris import HEIGHT, WIDTH

PAD = ("NEED_PIECE",)  # fed after a game's trace ends; those positions are not checked
GREEDY_PIECES = 100  # parameter: as in tests/test_isa.py, greedy games rarely top out


def expectations(trace):
    """(next token or None per position, board after each position)."""
    nxt = [trace[i + 1] if tok not in isa.STOP_TOKENS and i + 1 < len(trace) else None
           for i, tok in enumerate(trace)]
    board, boards = [0] * HEIGHT, []
    for tok in trace:
        if tok[0] == "WROW":
            board[tok[1]] = tok[2]
        boards.append(list(board))
    return nxt, boards


def decode_display(state):
    """[B, k_dim, v_dim] DISPLAY state -> [B, 20] row ints. Slot r holds L * dualrail(row r)."""
    on = state[:, :HEIGHT, 0 : 2 * WIDTH : 2]
    off = state[:, :HEIGHT, 1 : 2 * WIDTH : 2]
    weights = torch.tensor([1 << b for b in range(WIDTH)], device=state.device)
    return ((on > off).long() * weights).sum(-1)


def run_batch(model, prog, games):
    """games: [(name, trace)]. -> one result dict per game."""
    index = {t: i for i, t in enumerate(prog.vocab)}
    B, n = len(games), max(len(t) for _, t in games)
    ids = np.full((n, B), index[PAD], dtype=np.int64)
    exp_ids = np.full((n, B), -1, dtype=np.int64)
    boards = np.zeros((n, B, HEIGHT), dtype=np.int64)
    for b, (_, trace) in enumerate(games):
        nxt, bd = expectations(trace)
        ids[: len(trace), b] = [index[t] for t in trace]
        exp_ids[: len(trace), b] = [index[t] if t is not None else -1 for t in nxt]
        boards[: len(trace), b] = bd
        boards[len(trace) :, b] = bd[-1]
    run = BatchRunner(model, B)
    dev = run.device
    exp_t, boards_t = torch.as_tensor(exp_ids, device=dev), torch.as_tensor(boards, device=dev)
    lens = torch.as_tensor([len(t) for _, t in games], device=dev)
    first_tok_err = torch.full((B,), -1, device=dev, dtype=torch.long)
    first_board_err = torch.full((B,), -1, device=dev, dtype=torch.long)
    min_gap = torch.full((B,), float("inf"), device=dev)
    for i in range(n):
        got, gap, state = run.step(ids[i], prog.display_head)
        live = lens > i
        check = live & (exp_t[i] >= 0)
        bad = check & (got != exp_t[i]) & (first_tok_err < 0)
        first_tok_err[bad] = i
        min_gap[check] = torch.minimum(min_gap[check], gap[check])
        bbad = live & (decode_display(state) != boards_t[i]).any(-1) & (first_board_err < 0)
        first_board_err[bbad] = i
    out = []
    for b, (name, trace) in enumerate(games):
        out.append({
            "game": name,
            "tokens": len(trace),
            "checked": int((exp_ids[:, b] >= 0).sum()),
            "lines": sum(t[1] for t in trace if t[0] == "LINES"),
            "first_token_mismatch": int(first_tok_err[b]),
            "first_board_mismatch": int(first_board_err[b]),
            "min_logit_gap": float(min_gap[b]),
        })
    return out


def game_specs(n_random, n_greedy):
    specs = [(f"random-{s}", s, "random", None) for s in range(n_random)]
    specs += [(f"greedy-{s}", s, "greedy", GREEDY_PIECES) for s in range(n_greedy)]
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, required=True)
    ap.add_argument("--greedy", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--model", required=True, help="directory written by python -m deltris.build")
    ap.add_argument("--out", required=True, help="JSONL, one line per game, appended as batches finish")
    ap.add_argument("--device", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    from deltris.build import compile_deltris

    torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32: the circuit assumes fp32 identities
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    prog = compile_deltris()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    print(f"model {type(model).__name__} on {args.device}", flush=True)
    specs = game_specs(args.random, args.greedy)
    t0, done, fails = time.time(), 0, 0
    with open(args.out, "a") as fh:
        for k in range(0, len(specs), args.batch):
            chunk = specs[k : k + args.batch]
            games = [(name, play(seed, pol, max_pieces=mp)[0]) for name, seed, pol, mp in chunk]
            for r in run_batch(model, prog, games):
                fh.write(json.dumps(r) + "\n")
                done += 1
                fails += r["first_token_mismatch"] >= 0 or r["first_board_mismatch"] >= 0
            fh.flush()
            print(f"{done}/{len(specs)} games, {fails} failing, {time.time() - t0:.0f}s", flush=True)
    print(f"RESULT games={done} failing={fails} seconds={time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
