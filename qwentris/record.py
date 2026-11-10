"""Record a game played by the compiled model as an MP4 (SPEC §4 P3).

Panels: (1) the board decoded from the DISPLAY head's state, with the falling piece
decoded from the PIECE register's state; (2) the raw DISPLAY state matrix
[k_dim x v_dim] as a heatmap; (3) the registers decoded from state; (4) the scrolling
token trace with memory writes highlighted.

    python -m qwentris.record --model DIR --device cuda --out game.mp4

The host plays greedy moves planned from the decoded board. One video frame is drawn
at the end of every tick and after every token of the LOCK and CLEAR microprograms,
so line clears play back token by token.
"""

import argparse
import os
import pickle
import time

import numpy as np

from qwentris.ref import isa
from qwentris.ref.players import greedy_plan
from qwentris.ref.tetris import HEIGHT, WIDTH, SevenBag

FPS = 15  # parameter: one shown event per frame; 15/s keeps a falling piece readable
SECONDS = 60  # SPEC §4 P3: a 60-second recording
MAX_SEED = 2000  # parameter: seeds scanned for a multi-line clear early in the video
CLEAR_BY = 45  # parameter: the first multi-line clear must finish by this second
# parameter: pieces per isa game that pick_seed scans. It bounds the scan only; rollout()
# plays until SECONDS * FPS frames, not to this piece count.
SCAN_PIECES = 40


# ---- which tokens get a frame ---------------------------------------------------
def shown(trace):
    """Indices of trace tokens that get a frame: tick ends, and every token from a lock
    (SETP with lock=1) to the NEED_PIECE that follows it."""
    out, in_lock = [], False
    for i, tok in enumerate(trace):
        if tok[0] == "SETP" and tok[5] == 1:
            in_lock = True
        if in_lock or tok == ("TICK_END",) or tok == ("GAME_OVER",):
            out.append(i)
        if tok == ("NEED_PIECE",):
            in_lock = False
    return out


def pick_seed():
    """Lowest seed whose greedy game finishes a multi-line clear by CLEAR_BY seconds.
    Uses the isa interpreter only; P2 showed the model's trace equals it."""
    from qwentris.games import play

    for seed in range(MAX_SEED):
        trace, _ = play(seed, "greedy", max_pieces=SCAN_PIECES)
        idx = shown(trace)
        for f, i in enumerate(idx):
            if f / FPS > CLEAR_BY:
                break
            if trace[i][0] == "LINES" and trace[i][1] >= 2:
                end = next(g for g in range(f, len(idx)) if trace[idx[g]] == ("NEED_PIECE",))
                if end / FPS <= CLEAR_BY:
                    return seed, trace[i][1], end / FPS
    raise RuntimeError(f"no seed below {MAX_SEED} clears 2+ lines by {CLEAR_BY}s")


# ---- model rollout ----------------------------------------------------------------
def rollout(model, prog, seed, n_frames):
    """Play greedy Tetris on the model until n_frames frames are covered or game over."""
    from qwentris.host import Session

    sess = Session(model, prog)
    bag = SevenBag(seed)
    rec, frames = [], 0

    def take(steps):
        nonlocal frames
        for tok, gap, state in steps:
            regs = sess.registers(state)
            rec.append({"tok": tok, "gap": gap, "board": sess.board(state), "regs": regs,
                        "display": sess.display_matrix(state).astype(np.float32)})
        frames = len(shown([r["tok"] for r in rec]))

    take(sess.feed(("BOS",)))
    while frames < n_frames:
        p = bag.next()
        steps = sess.feed(("PIECE", p))
        take(steps)
        if steps[-1][0] == ("GAME_OVER",):
            break
        state = steps[-1][2]
        plan = greedy_plan(sess.board(state), sess.registers(state)["piece"])
        while frames < n_frames:
            steps = sess.feed(("ACT", plan.pop(0) if plan else "NOOP"))
            take(steps)
            if steps[-1][0] == ("NEED_PIECE",):
                break
    return rec


def check_against_isa(rec):
    """The recorded model trace must be a valid isa trace (same check as P2)."""
    trace = [r["tok"] for r in rec]
    mem, board = isa.Memory(), [0] * HEIGHT
    for i, tok in enumerate(trace):
        isa.write(tok, mem)
        if tok[0] == "WROW":
            board[tok[1]] = tok[2]
        assert rec[i]["board"] == board, f"board mismatch at token {i}"
        if tok not in isa.STOP_TOKENS and i + 1 < len(trace):
            assert isa.next_token(tok, mem) == trace[i + 1], f"trace diverges at token {i}"


# ---- rendering ----------------------------------------------------------------------
W, H = 1280, 720
BG, FG, DIM = (18, 18, 22), (230, 230, 230), (120, 120, 130)
PIECE_RGB = {"I": (80, 200, 230), "O": (235, 210, 60), "T": (170, 90, 210), "S": (90, 200, 90),
             "Z": (220, 70, 70), "J": (70, 110, 220), "L": (235, 150, 50)}
WRITE_RGB = {"WROW": (240, 120, 120), "SETP": (120, 200, 240), "SRC": (200, 170, 90),
             "PHASE": (170, 140, 240), "BOS": (200, 200, 200)}
HOST_RGB = (120, 220, 140)


def _fmt(tok):
    if tok[0] == "WROW":
        return f"WROW  r={tok[1]:<2d} {tok[2]:010b}"
    if tok[0] == "SETP":
        return f"SETP  {tok[1]} rot{tok[2]} x={tok[3]} y={tok[4]} lock={tok[5]}"
    return " ".join(str(t) for t in tok)


def render(rec, out, seconds):
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    big = ImageFont.load_default(size=22)
    trace = [r["tok"] for r in rec]
    idx = shown(trace)[: int(seconds * FPS)]
    writer = imageio_ffmpeg.write_frames(out, (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                         macro_block_size=8, quality=8)
    writer.send(None)
    lines = 0
    last_line_tok = 0
    cell = 30
    bx, by = 40, 90
    hx, hy, hc = 400, 110, 11  # heatmap origin and cell size
    for f, i in enumerate(idx):
        while last_line_tok <= i:
            if trace[last_line_tok][0] == "LINES":
                lines += trace[last_line_tok][1]
            last_line_tok += 1
        r = rec[i]
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((40, 20), "Qwentris: Tetris compiled into stock Qwen3.5 Gated DeltaNet weights", font=big, fill=FG)
        d.text((40, 52), f"token {i:5d}   lines {lines}   every pixel below is decoded from the model's "
                         "recurrent state", font=small, fill=DIM)

        # (1) board from DISPLAY state, falling piece from PIECE register state
        from qwentris.host import piece_overlay

        regs = r["regs"]
        live = set(piece_overlay(regs["piece"])) if regs["phase"] in ("PLAY", "LOCK") else set()
        color = PIECE_RGB.get(regs["piece"][0], FG)
        for row in range(HEIGHT):
            for c in range(WIDTH):
                x0, y0 = bx + c * cell, by + row * cell
                filled = r["board"][row] >> c & 1
                fill = (150, 150, 160) if filled else (35, 35, 42)
                if (row, c) in live:
                    fill = color
                d.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=fill)
        if trace[i][0] == "WROW":
            y0 = by + trace[i][1] * cell
            d.rectangle([bx - 3, y0 - 2, bx + WIDTH * cell, y0 + cell], outline=WRITE_RGB["WROW"], width=2)
        d.text((bx, by + HEIGHT * cell + 6), "board (DISPLAY head state)", font=small, fill=DIM)

        # (2) raw DISPLAY state matrix [k_dim x v_dim]
        M = r["display"]
        vmax = max(float(np.abs(M).max()), 1e-6)
        for k in range(M.shape[0]):
            for v in range(M.shape[1]):
                g = int(255 * max(0.0, float(M[k, v])) / vmax)
                d.rectangle([hx + v * hc, hy + k * hc, hx + v * hc + hc - 1, hy + k * hc + hc - 1],
                            fill=(g, int(g * 0.8), int(g * 0.3)))
        if trace[i][0] == "WROW":
            y0 = hy + trace[i][1] * hc
            d.rectangle([hx - 2, y0 - 1, hx + M.shape[1] * hc + 1, y0 + hc], outline=WRITE_RGB["WROW"], width=2)
        d.text((hx, hy - 20), f"DISPLAY head state, raw [{M.shape[0]} keys x {M.shape[1]} values]",
               font=small, fill=DIM)
        d.text((hx, hy + M.shape[0] * hc + 4),
               "row r = key slot r; columns alternate on/off rail per board column", font=small, fill=DIM)

        # (3) registers from state
        ry = hy + M.shape[0] * hc + 40
        p = regs["piece"]
        for j, (k, v) in enumerate([("PHASE", regs["phase"]), ("T", regs["T"]), ("S", regs["S"]),
                                    ("PIECE", f"{p[0]} rot{p[1]} x={p[2]} y={p[3]}")]):
            d.text((hx, ry + j * 24), f"{k:<6s}", font=font, fill=DIM)
            d.text((hx + 70, ry + j * 24), str(v), font=font, fill=FG)
        d.text((hx, ry + 4 * 24 + 4), "registers (single-slot heads, decoded from state)", font=small, fill=DIM)

        # (4) token trace
        tx, ty, n = 940, 90, 30
        d.text((tx, ty - 22), "token trace", font=small, fill=DIM)
        first = max(0, i - n + 1)
        for j, t in enumerate(trace[first : i + 1]):
            rgb = HOST_RGB if t[0] in ("PIECE", "ACT") else WRITE_RGB.get(t[0], DIM)
            if first + j == i:
                d.rectangle([tx - 4, ty + j * 19 - 1, W - 20, ty + j * 19 + 17], fill=(45, 45, 55))
            d.text((tx, ty + j * 19), _fmt(t), font=small, fill=rgb)
        ly = ty + n * 19 + 8
        d.text((tx, ly), "green: host   colored: memory write", font=small, fill=DIM)
        writer.send(np.asarray(img).tobytes())
    writer.close()
    return len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", required=True, help="MP4 path; the rollout is saved beside it as .pkl")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    from qwentris.build import compile_qwentris

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed, k, at = pick_seed()
    print(f"seed {seed}: first {k}-line clear done at {at:.1f}s of video", flush=True)
    prog = compile_qwentris()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    t0 = time.time()
    rec = rollout(model, prog, seed, SECONDS * FPS)
    print(f"rollout {len(rec)} tokens in {time.time() - t0:.0f}s, min gap {min(r['gap'] for r in rec):.3f}",
          flush=True)
    check_against_isa(rec)
    print("rollout equals the isa trace and oracle board", flush=True)
    with open(os.path.splitext(args.out)[0] + ".pkl", "wb") as fh:
        pickle.dump({"seed": seed, "rec": rec}, fh)
    t0 = time.time()
    n = render(rec, args.out, SECONDS)
    last = shown([r["tok"] for r in rec])[n - 1]
    multi = [r["tok"] for r in rec[: last + 1] if r["tok"][0] == "LINES" and r["tok"][1] >= 2]
    print(f"RESULT frames={n} seconds={n / FPS:.1f} multi_line_clears={len(multi)} "
          f"render={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
