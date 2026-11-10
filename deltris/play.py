"""Play Deltris in the terminal (SPEC §4 P3).

    uv run python -m deltris.play --model models/deltris --device cpu

Arrow keys: left/right move, up rotates clockwise, down (or no key within
TICK_SECONDS) sends NOOP. q quits. The board, the falling piece and the counters are
drawn only from the model's decoded DeltaNet state; this file holds no game state
beyond the piece bag, which is the host's input stream.
"""

import argparse
import curses
import time

from deltris.ref.tetris import HEIGHT, WIDTH, SevenBag

TICK_SECONDS = 0.5  # parameter: wait for a key before sending NOOP; gravity runs once per ACT
KEYS = {curses.KEY_LEFT: "LEFT", curses.KEY_RIGHT: "RIGHT", curses.KEY_UP: "ROT_CW", curses.KEY_DOWN: "NOOP"}


def run(sess, bag, next_action, draw):
    """Host loop. next_action() -> action name or None to quit; draw(state, info) shows a frame.
    -> number of lines cleared, counted from the model's LINES tokens."""
    from deltris.ref import isa

    lines, ms = 0, 0.0

    def feed(tok):
        nonlocal lines, ms
        t = time.time()
        steps = sess.feed(tok)
        ms = (time.time() - t) * 1e3 / len(steps)
        lines += sum(s[0][1] for s in steps if s[0][0] == "LINES")
        last = steps[-1]
        draw(last[2], {"lines": lines, "ms": ms, "last": last[0]})
        return last[0]

    stop = feed(("BOS",))
    while True:
        if stop == ("GAME_OVER",):
            return lines
        if stop == ("NEED_PIECE",):
            stop = feed(("PIECE", bag.next()))
            continue
        assert stop in isa.STOP_TOKENS, stop
        a = next_action()
        if a is None:
            return lines
        stop = feed(("ACT", a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--seed", type=int, default=0, help="piece bag seed")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    from deltris.build import compile_deltris
    from deltris.host import Session, piece_overlay

    prog = compile_deltris()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device).eval()
    sess = Session(model, prog)

    def ui(scr):
        curses.curs_set(0)
        scr.timeout(int(TICK_SECONDS * 1000))

        def next_action():
            k = scr.getch()
            if k in (ord("q"), ord("Q")):
                return None
            return KEYS.get(k, "NOOP")

        def draw(state, info):
            board, regs = sess.board(state), sess.registers(state)
            live = set(piece_overlay(regs["piece"])) if regs["phase"] in ("PLAY", "LOCK") else set()
            scr.erase()
            scr.addstr(0, 0, "Deltris  (drawn from the model's DeltaNet state)")
            for r in range(HEIGHT):
                row = "".join("[]" if (r, c) in live else ("##" if board[r] >> c & 1 else " .")
                              for c in range(WIDTH))
                scr.addstr(2 + r, 2, "|" + row + "|")
            scr.addstr(2 + HEIGHT, 2, "+" + "--" * WIDTH + "+")
            col = 2 * WIDTH + 6
            p = regs["piece"]
            for j, s in enumerate([f"phase  {regs['phase']}", f"piece  {p[0]} rot{p[1]} x={p[2]} y={p[3]}",
                                   f"lines  {info['lines']}", f"last   {' '.join(map(str, info['last']))}",
                                   f"{info['ms']:.0f} ms/token", "", "<- -> move  ^ rotate", "v next tick  q quit"]):
                scr.addstr(3 + j, col, s)
            if info["last"] == ("GAME_OVER",):
                scr.addstr(3 + 9, col, "GAME OVER (any key)")
                scr.timeout(-1)
                scr.getch()
            scr.refresh()

        return run(sess, SevenBag(args.seed), next_action, draw)

    lines = curses.wrapper(ui)
    print(f"lines cleared: {lines}")


if __name__ == "__main__":
    main()
