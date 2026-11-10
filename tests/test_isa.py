"""P0 acceptance: the protocol interpreter (isa.py) matches the engine (tetris.py)."""

import os
import random

import pytest

from deltris.ref import isa
from deltris.ref.players import greedy_plan
from deltris.ref.tetris import ACTIONS, FULL_ROW, Game, SevenBag

N_GAMES = int(os.environ.get("DELTRIS_P0_GAMES", "10000"))  # parameter: SPEC §4 P0 asks for 10 000
# parameter: random play almost never clears lines (1 line in 50 games), so a
# second set of greedy games covers CLEAR. 500 x 100 pieces gives ~18k lines and
# ~1.3k multi-line clears in ~25 s on one core.
N_GREEDY = int(os.environ.get("DELTRIS_P0_GREEDY", "500"))
GREEDY_PIECES = 100


def check_state(g, m):
    assert m.mem.rows == g.board
    assert m.lines == g.lines
    if g.piece is not None:
        assert m.mem.piece == g.piece


def run_paired(g, m, pieces, actions, max_ticks=None, on_spawn=None):
    """Drive engine and interpreter in lockstep. Returns the stop token that ended the game or run."""
    ticks = 0
    for p in pieces:
        ok = g.spawn(p)
        if ok and on_spawn is not None:
            on_spawn(g)
        out = m.feed(("PIECE", p))
        assert out == (("TICK_END",) if ok else ("GAME_OVER",))
        if not ok:
            return out
        check_state(g, m)
        while True:
            a = next(actions)
            res = g.tick(a)
            out = m.feed(("ACT", a))
            assert out == (("TICK_END",) if res == "moved" else ("NEED_PIECE",))
            check_state(g, m)
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                return out
            if res == "locked":
                break
    return out


def booted(board=None):
    g, m = Game(), isa.Machine()
    assert m.feed(("BOS",)) == ("NEED_PIECE",)
    if board is not None:  # test fixture: seed both sides with the same board
        g.board = list(board)
        m.mem.rows = list(board)
    return g, m


def scripted(seq):
    it = iter(seq)
    while True:
        yield next(it, "NOOP")


def piece_stream(bag):
    while True:
        yield bag.next()


def test_random_games_match_engine():
    lines = 0
    for seed in range(N_GAMES):
        rng = random.Random(seed)
        g, m = booted()
        actions = iter(lambda: rng.choice(ACTIONS), None)
        assert run_paired(g, m, piece_stream(SevenBag(seed)), actions) == ("GAME_OVER",)
        lines += g.lines
    print(f"\n{N_GAMES} random games to top-out, {lines} lines")


def test_greedy_games_match_engine():
    lines = multi = 0
    for seed in range(N_GREEDY):
        g, m = booted()
        plan = []
        stream = piece_stream(SevenBag(seed))
        run_paired(
            g,
            m,
            (next(stream) for _ in range(GREEDY_PIECES)),
            iter(lambda: plan.pop(0) if plan else "NOOP", None),
            on_spawn=lambda g: plan.__init__(greedy_plan(g.board, g.piece)),
        )
        lines += g.lines
        multi += sum(1 for t in m.trace if t[0] == "LINES" and t[1] >= 2)
    print(f"\n{N_GREEDY} greedy games x {GREEDY_PIECES} pieces, {lines} lines, {multi} multi-line clears")


def test_tetris_four_line_clear():
    board = [0] * 20
    for r in range(16, 20):
        board[r] = FULL_ROW & ~(1 << 9)
    board[15] = 0b11  # residue above the stack must fall by four
    g, m = booted(board)
    run_paired(g, m, ["I"], scripted(["ROT_CW", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]))
    assert ("LINES", 4) in m.trace
    assert g.board[19] == 0b11 and sum(g.board) == 0b11


def test_non_contiguous_clear():
    board = [0] * 20
    board[17] = FULL_ROW & ~1
    board[18] = FULL_ROW & ~1 & ~(1 << 5)
    board[19] = FULL_ROW & ~1
    g, m = booted(board)
    run_paired(g, m, ["I"], scripted(["ROT_CW"] + ["LEFT"] * 5))
    assert ("LINES", 2) in m.trace
    assert g.board[19] == FULL_ROW & ~(1 << 5)
    assert g.board[18] == 1


def test_top_out():
    board = [0] * 20
    board[1] = 1 << 4
    g, m = booted(board)
    assert run_paired(g, m, ["T"], scripted([])) == ("GAME_OVER",)
    assert m.trace[-2:] == [("PHASE", "OVER"), ("GAME_OVER",)]


def test_rotation_rejected_at_wall():
    g, m = booted()
    run_paired(g, m, ["I"], scripted(["ROT_CW"] + ["RIGHT"] * 4 + ["ROT_CW"]), max_ticks=6)
    assert m.mem.piece[1] == 1 and m.mem.piece[2] == 7


def test_lock_after_legal_action_uses_post_action_position():
    # The piece slides right on the tick it locks; the lock must use the slid position.
    board = [0] * 20
    g, m = booted(board)
    run_paired(g, m, ["O"], scripted(["NOOP"] * 18 + ["RIGHT"]), max_ticks=19)
    assert g.board[18] == g.board[19] == 0b11 << 5


def test_unwritten_read_raises():
    with pytest.raises(isa.UnwrittenRead):
        isa.next_token(("ACT", "NOOP"), isa.Memory())


def test_vocabulary_unique():
    v = isa.vocabulary()
    assert len(v) == len(set(v))
    print(f"\nvocabulary size {len(v)}")
