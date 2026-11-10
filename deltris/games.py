"""Game traces from the protocol interpreter, for differential tests and recording."""

import random

from deltris.ref import isa
from deltris.ref.players import greedy_plan
from deltris.ref.tetris import ACTIONS, Game, SevenBag


def play(seed, policy="random", max_pieces=None):
    """Full isa trace of one game. Returns (trace, set of trace indices fed by the host)."""
    rng = random.Random(seed)
    bag = SevenBag(seed)
    g, m = Game(), isa.Machine()
    host = {0}
    m.feed(("BOS",))
    pieces = 0
    while max_pieces is None or pieces < max_pieces:
        p = bag.next()
        pieces += 1
        ok = g.spawn(p)
        host.add(len(m.trace))
        m.feed(("PIECE", p))
        if not ok:
            break
        plan = greedy_plan(g.board, g.piece) if policy == "greedy" else None
        while True:
            a = (plan.pop(0) if plan else "NOOP") if plan is not None else rng.choice(ACTIONS)
            res = g.tick(a)
            host.add(len(m.trace))
            m.feed(("ACT", a))
            if res == "locked":
                break
    return m.trace, host
