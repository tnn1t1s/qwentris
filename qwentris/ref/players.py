"""Host-side action sources for tests. Random play almost never clears lines, so
the greedy player exists to exercise the CLEAR microprogram in bulk."""

from qwentris.ref.tetris import HEIGHT, WIDTH, X_RANGE, clear_full_rows, legal, piece_cells


def _drop(board, piece):
    p, rot, x, y = piece
    while legal(board, (p, rot, x, y + 1)):
        y += 1
    return (p, rot, x, y)


def _score(board, piece):
    b = list(board)
    for r, c in piece_cells(piece):
        b[r] |= 1 << c
    b, removed = clear_full_rows(b)
    heights = [0] * WIDTH
    covered = 0
    holes = 0
    for r, row in enumerate(b):
        holes += bin(covered & ~row).count("1")
        new = row & ~covered
        while new:
            low = new & -new
            heights[low.bit_length() - 1] = HEIGHT - r
            new ^= low
        covered |= row
    bump = sum(abs(heights[i] - heights[i + 1]) for i in range(WIDTH - 1))
    # heuristic: weights from the common 4-feature hand-tuned Tetris evaluator; only need "clears lines often"
    return 0.76 * removed - 0.51 * sum(heights) - 0.36 * holes - 0.18 * bump


def greedy_plan(board, piece):
    """Action list that rotates, then shifts, toward the best drop target.

    Approximate: it ignores gravity between moves, so the engine may reject some
    actions. That is fine; it only needs to produce many line clears."""
    p, _, _, _ = piece
    best = None
    for k in range(4):
        rot = (piece[1] + k) % 4
        for x in X_RANGE:
            if not legal(board, (p, rot, x, piece[3])):
                continue
            s = _score(board, _drop(board, (p, rot, x, piece[3])))
            if best is None or s > best[0]:
                best = (s, k, x - piece[2])
    if best is None:
        return []
    _, k, dx = best
    return ["ROT_CW"] * k + (["RIGHT"] * dx if dx > 0 else ["LEFT"] * -dx)
