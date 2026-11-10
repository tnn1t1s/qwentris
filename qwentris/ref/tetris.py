"""Reference Tetris engine (SPEC §2). The oracle for every other component.

Board: 10 columns x 20 rows, row 0 at the top. A row is an int; bit c is column c.
Piece position (x, y) is the top-left of its 4x4 box.
"""

import random

WIDTH = 10  # invariant: SPEC §2
HEIGHT = 20  # invariant: SPEC §2
FULL_ROW = (1 << WIDTH) - 1

PIECES = "IOTSZJL"
ACTIONS = ("LEFT", "RIGHT", "ROT_CW", "NOOP")

X_RANGE = range(-2, 10)  # invariant: SPEC §2, x in [-2, 9]
Y_RANGE = range(-2, 20)  # invariant: SPEC §2, y in [-2, 19]

# Rotation-0 shapes as (row, col) inside the box, plus the box size used for
# clockwise rotation. SRS shapes; O sits at cols 0-1 so spawn x=4 puts it in
# board columns 4-5 (see DECISIONS.md).
_BASE = {
    "I": (4, ((1, 0), (1, 1), (1, 2), (1, 3))),
    "O": (2, ((0, 0), (0, 1), (1, 0), (1, 1))),
    "T": (3, ((0, 1), (1, 0), (1, 1), (1, 2))),
    "S": (3, ((0, 1), (0, 2), (1, 0), (1, 1))),
    "Z": (3, ((0, 0), (0, 1), (1, 1), (1, 2))),
    "J": (3, ((0, 0), (1, 0), (1, 1), (1, 2))),
    "L": (3, ((0, 2), (1, 0), (1, 1), (1, 2))),
}


def _rotations(n, cells):
    out = [frozenset(cells)]
    for _ in range(3):
        out.append(frozenset((c, n - 1 - r) for r, c in out[-1]))
    return tuple(out)


SHAPES = {p: _rotations(*_BASE[p]) for p in PIECES}


def spawn_position(p):
    return (p, 0, 4 if p == "O" else 3, 0)


def piece_cells(piece):
    p, rot, x, y = piece
    return [(y + r, x + c) for r, c in SHAPES[p][rot]]


def piece_row_mask(piece, row):
    """Bits the piece occupies in board row `row` (cells must be in bounds)."""
    mask = 0
    for r, c in piece_cells(piece):
        if r == row:
            mask |= 1 << c
    return mask


def legal(board, piece):
    for r, c in piece_cells(piece):
        if not (0 <= r < HEIGHT and 0 <= c < WIDTH):
            return False
        if board[r] >> c & 1:
            return False
    return True


def apply_action(piece, action):
    p, rot, x, y = piece
    if action == "LEFT":
        return (p, rot, x - 1, y)
    if action == "RIGHT":
        return (p, rot, x + 1, y)
    if action == "ROT_CW":
        return (p, (rot + 1) % 4, x, y)
    return piece


def clear_full_rows(board):
    kept = [row for row in board if row != FULL_ROW]
    removed = HEIGHT - len(kept)
    return [0] * removed + kept, removed


class Game:
    def __init__(self):
        self.board = [0] * HEIGHT
        self.piece = None
        self.lines = 0
        self.over = False

    def spawn(self, p):
        """Spawn piece type p. Returns False (and ends the game) if illegal."""
        piece = spawn_position(p)
        if not legal(self.board, piece):
            self.over = True
            return False
        self.piece = piece
        return True

    def tick(self, action):
        """Apply action if legal, then gravity. Returns 'moved' or 'locked'."""
        cand = apply_action(self.piece, action)
        if legal(self.board, cand):
            self.piece = cand
        p, rot, x, y = self.piece
        down = (p, rot, x, y + 1)
        if legal(self.board, down):
            self.piece = down
            return "moved"
        for r, c in piece_cells(self.piece):
            self.board[r] |= 1 << c
        self.board, removed = clear_full_rows(self.board)
        self.lines += removed
        self.piece = None
        return "locked"


class SevenBag:
    """Seeded 7-bag piece source. Host-side only; the model has no RNG."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.bag = []

    def next(self):
        if not self.bag:
            self.bag = list(PIECES)
            self.rng.shuffle(self.bag)
        return self.bag.pop()
