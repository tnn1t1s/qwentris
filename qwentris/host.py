"""Host loop over the compiled model, and decoding of the model's recurrent state.

The host appends PIECE and ACT tokens and the model emits tokens until a stop token
(SPEC §3). Everything shown about the game (board, falling piece, phase, counters)
is decoded from the layer-0 DeltaNet state, never from Python game state.
"""

import numpy as np
import torch
from transformers import DynamicCache

from qwentris.ref import isa
from qwentris.ref.tetris import HEIGHT, WIDTH, piece_cells


class Session:
    def __init__(self, model, prog):
        self.model, self.prog = model, prog
        self.device = next(model.parameters()).device
        self.index = {t: i for i, t in enumerate(prog.vocab)}
        self.cache = DynamicCache(config=model.config)
        self.pos = 0
        self.heads = {h.name: (i, h) for i, h in enumerate(prog.heads)}

    @torch.no_grad()
    def step(self, tok):
        """Feed one token. -> (predicted next token, logit gap, layer-0 state [H, k, v] numpy)."""
        ids = torch.tensor([[self.index[tok]]], device=self.device)
        pos = torch.tensor([[self.pos]], device=self.device)
        self.pos += 1
        out = self.model(input_ids=ids, position_ids=pos, attention_mask={"linear_attention": None},
                         past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        self.cache = out.past_key_values
        top = out.logits[0, -1].topk(2)
        # observation: on CPU, .cpu().numpy() is a view of the cache tensor, which the next
        # step updates in place; the copy keeps this token's state.
        state = self.cache.get_recurrent_state(0, state_idx=0)[0].float().cpu().numpy().copy()
        return self.prog.vocab[int(top.indices[0])], float(top.values[0] - top.values[1]), state

    def feed(self, host_tok):
        """Feed a host token, then the model's own tokens until a stop token.
        -> list of (token, logit gap of the prediction made after it, state after it)."""
        out, tok = [], host_tok
        while True:
            nxt, gap, state = self.step(tok)
            out.append((tok, gap, state))
            if tok in isa.STOP_TOKENS:
                return out
            tok = nxt

    # ---- decoding the state ---------------------------------------------------
    def _value(self, state, name, slot=0):
        i, h = self.heads[name]
        return state[i, slot, : len(h.value_dims)], h

    def _fields(self, state, name):
        vec, h = self._value(state, name)
        vals, j = [], 0
        for f in h.write_values:
            seg = vec[j : j + len(f.dims)]
            j += len(f.dims)
            if f.kind == "onehot":
                vals.append(f.labels[int(np.argmax(seg))] if seg.max() > 0 else None)
            else:
                vals.append(sum(1 << b for b in range(f.nbits) if seg[2 * b] > seg[2 * b + 1]))
        return vals

    def board(self, state):
        """20 row ints from the DISPLAY head: slot r holds L * dualrail(row r)."""
        i, _ = self.heads["DISPLAY"]
        S = state[i]
        return [sum(1 << b for b in range(WIDTH) if S[r, 2 * b] > S[r, 2 * b + 1]) for r in range(HEIGHT)]

    def display_matrix(self, state):
        i, _ = self.heads["DISPLAY"]
        return state[i]

    def registers(self, state):
        piece = tuple(self._fields(state, "PIECE"))
        (phase,) = self._fields(state, "PHASE")
        (t,) = self._fields(state, "T")
        (s,) = self._fields(state, "S")
        return {"piece": piece, "phase": phase, "T": t, "S": s}


def piece_overlay(piece):
    """Board cells of a decoded PIECE register value, clipped to the board."""
    if None in piece:
        return []
    return [(r, c) for r, c in piece_cells(piece) if 0 <= r < HEIGHT and 0 <= c < WIDTH]
