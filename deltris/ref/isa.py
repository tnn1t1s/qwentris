"""Interpreter for the Deltris token protocol (SPEC §3, as amended in DECISIONS.md).

The interpreter is held to the same information limit as the compiled model:
`next_token(tok, mem)` sees only the current token and the memory map after that
token's write. It has no trace history and no host game state. Every write is a
pure function of the token (`write`). Reading an unwritten memory slot raises,
because the model's RMSNormGated read of an unwritten slot returns zero and breaks
the constant-norm rule (SPEC §3.2).
"""

from dataclasses import dataclass, field

from deltris.ref.tetris import (
    ACTIONS,
    FULL_ROW,
    HEIGHT,
    PIECES,
    X_RANGE,
    Y_RANGE,
    apply_action,
    legal,
    piece_cells,
    piece_row_mask,
    spawn_position,
)

PHASES = ("SPAWN", "PLAY", "LOCK", "CLEAR", "OVER")
T_RANGE = range(-1, HEIGHT + 1)  # -1 = before the first LOCK row; 20 = before compaction
S_RANGE = range(-1, HEIGHT)  # -1 = no source row left
STOP_TOKENS = {("TICK_END",), ("NEED_PIECE",), ("GAME_OVER",)}

# Placeholder PIECE register value written by BOS. Never used for game logic:
# PIECE is always overwritten by SETP before PLAY.
BOOT_PIECE = spawn_position("I")


def vocabulary():
    """Every token, in id order. Tokens are tuples."""
    v = [("BOS",)]
    v += [("ACT", a) for a in ACTIONS]
    v += [("PIECE", p) for p in PIECES]
    v += [("WROW", r, b) for r in range(HEIGHT) for b in range(1 << 10)]
    v += [
        ("SETP", p, rot, x, y, lock)
        for p in PIECES
        for rot in range(4)
        for x in X_RANGE
        for y in Y_RANGE
        for lock in (0, 1)
    ]
    v += [("SRC", s) for s in S_RANGE]
    v += [("PHASE", p) for p in PHASES]
    v += [("TICK_END",), ("NEED_PIECE",), ("GAME_OVER",)]
    v += [("LINES", k) for k in range(1, 5)]
    return v


class UnwrittenRead(RuntimeError):
    pass


@dataclass
class Memory:
    rows: list = field(default_factory=lambda: [None] * HEIGHT)
    piece: tuple = None
    phase: str = None
    T: int = None
    S: int = None

    def check_written(self):
        if any(r is None for r in self.rows) or None in (self.piece, self.phase, self.T, self.S):
            raise UnwrittenRead(f"read of unwritten slot: {self}")


def write(tok, mem):
    """Apply the memory write that token `tok` encodes."""
    kind = tok[0]
    if kind == "BOS":
        mem.rows = [0] * HEIGHT
        mem.piece = BOOT_PIECE
        mem.phase = "SPAWN"
        mem.T = -1
        mem.S = -1
    elif kind == "WROW":
        mem.rows[tok[1]] = tok[2]
        mem.T = tok[1]
    elif kind == "SETP":
        mem.piece = tok[1:5]
    elif kind == "SRC":
        mem.S = tok[1]
    elif kind == "PHASE":
        mem.phase = tok[1]
        if tok[1] == "LOCK":
            mem.T = -1
        elif tok[1] == "CLEAR":
            mem.T = HEIGHT


def _piece_rows(piece):
    return sorted({r for r, _ in piece_cells(piece)})


def _first_nonfull_at_or_above(rows, start):
    for r in range(start, -1, -1):
        if rows[r] != FULL_ROW:
            return r
    return -1


def next_token(tok, mem):
    """The token the model must emit after `tok`, given memory after `tok`'s write."""
    mem.check_written()
    kind, phase, rows = tok[0], mem.phase, mem.rows

    if kind == "BOS":
        return ("NEED_PIECE",)

    if kind == "PIECE":
        piece = spawn_position(tok[1])
        if legal(rows, piece):
            return ("SETP", *piece, 0)
        return ("PHASE", "OVER")

    if kind == "SETP":
        if tok[5] == 1:
            return ("PHASE", "LOCK")
        return ("PHASE", "PLAY") if phase == "SPAWN" else ("TICK_END",)

    if kind == "ACT":
        cand = apply_action(mem.piece, tok[1])
        pos = cand if legal(rows, cand) else mem.piece
        p, rot, x, y = pos
        if legal(rows, (p, rot, x, y + 1)):
            return ("SETP", p, rot, x, y + 1, 0)
        return ("SETP", *pos, 1)

    if kind == "PHASE":
        p = tok[1]
        if p == "PLAY":
            return ("TICK_END",)
        if p == "OVER":
            return ("GAME_OVER",)
        if p == "SPAWN":
            return ("NEED_PIECE",)
        if p == "LOCK":
            return _next_lock_row(mem)
        return ("LINES", sum(r == FULL_ROW for r in mem.rows))  # CLEAR

    if kind == "LINES":
        return ("SRC", _first_nonfull_at_or_above(rows, HEIGHT - 1))

    if kind == "SRC":
        t = mem.T - 1
        return ("WROW", t, rows[mem.S] if mem.S >= 0 else 0)

    if kind == "WROW" and phase == "LOCK":
        return _next_lock_row(mem)

    if kind == "WROW" and phase == "CLEAR":
        if mem.T == 0:
            return ("PHASE", "SPAWN")
        s = _first_nonfull_at_or_above(rows, mem.S - 1) if mem.S >= 0 else -1
        return ("SRC", s)

    raise ValueError(f"no transition for {tok} in phase {phase}")


def _next_lock_row(mem):
    for r in _piece_rows(mem.piece):
        if r > mem.T:
            return ("WROW", r, mem.rows[r] | piece_row_mask(mem.piece, r))
    if any(r == FULL_ROW for r in mem.rows):
        return ("PHASE", "CLEAR")
    return ("PHASE", "SPAWN")


class Machine:
    """Runs the protocol: the host appends a token, the machine emits until a stop token."""

    def __init__(self):
        self.mem = Memory()
        self.trace = []
        self.lines = 0

    def feed(self, host_tok):
        tok = host_tok
        while True:
            self.trace.append(tok)
            write(tok, self.mem)
            if tok[0] == "LINES":
                self.lines += tok[1]
            if tok in STOP_TOKENS:
                return tok
            tok = next_token(tok, self.mem)
