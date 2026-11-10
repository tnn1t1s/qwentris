from qwentris.ref.tetris import (
    FULL_ROW,
    PIECES,
    SHAPES,
    Game,
    SevenBag,
    apply_action,
    clear_full_rows,
    legal,
    piece_row_mask,
    spawn_position,
)


def grid(piece_type, rot):
    cells = SHAPES[piece_type][rot]
    return ["".join("X" if (r, c) in cells else "." for c in range(4)) for r in range(4)]


def test_srs_shapes_hand_checked():
    assert grid("T", 0)[:2] == [".X..", "XXX."]
    assert grid("T", 1)[:3] == [".X..", ".XX.", ".X.."]
    assert grid("T", 2)[:3] == ["....", "XXX.", ".X.."]
    assert grid("T", 3)[:3] == [".X..", "XX..", ".X.."]
    assert grid("I", 0)[1] == "XXXX"
    assert [row[2] for row in grid("I", 1)] == ["X"] * 4
    assert grid("I", 2)[2] == "XXXX"
    assert [row[1] for row in grid("I", 3)] == ["X"] * 4
    assert grid("S", 0)[:2] == [".XX.", "XX.."]
    assert grid("Z", 0)[:2] == ["XX..", ".XX."]
    assert grid("J", 0)[:2] == ["X...", "XXX."]
    assert grid("L", 0)[:2] == ["..X.", "XXX."]
    for rot in range(4):
        assert grid("O", rot)[:2] == ["XX..", "XX.."]
    for p in PIECES:
        assert all(len(s) == 4 for s in SHAPES[p])


def test_spawn_columns():
    assert spawn_position("O") == ("O", 0, 4, 0)
    assert spawn_position("T") == ("T", 0, 3, 0)


def test_legal_bounds_and_collision():
    board = [0] * 20
    assert legal(board, ("I", 0, 6, 0))
    assert not legal(board, ("I", 0, 7, 0))  # right wall
    assert not legal(board, ("I", 0, -1, 0))  # left wall
    assert not legal(board, ("I", 1, 0, 17))  # floor
    assert not legal(board, ("T", 0, 3, -1))  # ceiling
    board[1] = 1 << 4
    assert not legal(board, ("T", 0, 3, 0))  # occupied cell


def test_apply_action():
    p = ("T", 3, 4, 5)
    assert apply_action(p, "LEFT") == ("T", 3, 3, 5)
    assert apply_action(p, "RIGHT") == ("T", 3, 5, 5)
    assert apply_action(p, "ROT_CW") == ("T", 0, 4, 5)
    assert apply_action(p, "NOOP") == p


def test_piece_row_mask():
    assert piece_row_mask(("T", 0, 3, 0), 0) == 1 << 4
    assert piece_row_mask(("T", 0, 3, 0), 1) == 0b111 << 3
    assert piece_row_mask(("T", 0, 3, 0), 2) == 0


def test_clear_full_rows():
    board = [0] * 20
    board[17] = FULL_ROW
    board[18] = 5
    board[19] = FULL_ROW
    new, removed = clear_full_rows(board)
    assert removed == 2
    assert new[19] == 5 and sum(new) == 5


def test_tick_move_reject_and_lock():
    g = Game()
    assert g.spawn("I")
    assert g.tick("ROT_CW") == "moved"
    assert g.piece == ("I", 1, 3, 1)
    for _ in range(4):
        g.tick("RIGHT")
    assert g.piece[2] == 7  # vertical I in column 9
    g.tick("ROT_CW")  # rot 2 would cover columns 7..10: rejected
    assert g.piece[1] == 1
    while g.tick("NOOP") == "moved":
        pass
    assert g.piece is None
    assert g.board[16:] == [1 << 9] * 4


def test_lock_clears_lines():
    g = Game()
    g.board[19] = FULL_ROW & ~(0b1111 << 3)
    g.spawn("I")
    while g.tick("NOOP") == "moved":
        pass
    assert g.lines == 1 and g.board == [0] * 20


def test_game_over():
    g = Game()
    g.board[1] = 1 << 4
    assert not g.spawn("T")
    assert g.over


def test_seven_bag():
    bag = SevenBag(0)
    for _ in range(3):
        assert sorted(bag.next() for _ in range(7)) == sorted(PIECES)
