"""Program graph: fields, MLP units, memory heads, embedding and LM head.

Stages order the residual stream. The embedding is stage 0. In layer i the mixer
writes at stage 2i+1 and the MLP at stage 2i+2. An op runs in the earliest
layer whose input stage it can see, so the layer count is the circuit depth.

Each op also registers a reference function over the ideal residual vector
(every dim exactly 0 or 1). `Program.reference` runs those functions, giving the
NumPy reference that the compiled HF model is tested against.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from deltris.compiler.layout import Field

# parameter: SiLU gain for MLP units. Chosen so silu(G)/G == 1.0 and |silu(-G)| < 1e-11
# in fp32 (tests/test_primitives.py checks both); larger values buy nothing.
G = 30.0
# parameter: drive level for memory q/k/v, the beta logit and the RMSNormGated gate z.
# SPEC §1: sigmoid(L) == 1.0 and silu(L) == L in fp32.
L = 30.0
# parameter: A_log for every DeltaNet head; exp(-exp(A_LOG) * softplus(.)) == 1.0 in fp32 (SPEC §1).
A_LOG = -40.0
# parameter: logit gap between the correct token and the best wrong token. 30 makes the
# softmax one-hot in fp32; greedy decoding only needs the gap to be positive.
LOGIT_SCALE = 30.0
# invariant: transformers Qwen3_5 default rms_norm_eps; the compiler folds it into every scale.
RMS_EPS = 1e-6


@dataclass
class Unit:
    """One MLP hidden unit: down += silu(gate . x) * (up . x). Weights on ideal residual values."""

    gate: dict
    up: dict
    down: dict


@dataclass
class Head:
    """One memory head in the layer-0 mixer. Single or multi slot.

    k = L * e_j when the token's write_slot field holds slot j; k = 0 if absent.
    v = L * (stored pattern: the concatenated dims of write_values). q = L * e_read_slot.
    The read goes to `outs`, one field per value field.
    """

    name: str
    n_slots: int
    write_slot: Field  # optional onehot over range(n_slots), stage 0
    write_values: list  # fields whose dims are stored, stage 0
    read_slot: int
    outs: list | None  # None: state is kept but never read into the residual

    @property
    def value_dims(self):
        return [d for f in self.write_values for d in f.dims]

    @property
    def out_dims(self):
        return [d for f in self.outs for d in f.dims]

    @property
    def value_count(self):
        return sum(f.count for f in self.write_values)


@dataclass
class Program:
    ndims: int = 0
    fields: list = field(default_factory=list)
    units: dict = field(default_factory=lambda: defaultdict(list))
    heads: list = field(default_factory=list)
    refs: list = field(default_factory=list)  # (kind, fn) in topological order
    _const_unit: dict = field(default_factory=dict)

    def __post_init__(self):
        self.one = self._alloc("ONE", "one", (), 0, n=1)
        self.vocab = None
        self.embed_fn = None
        self.emit_fn = None

    # ---- allocation -------------------------------------------------------
    def _alloc(self, name, kind, labels, stage, n=None, optional=False):
        if hasattr(self, "_fill"):
            raise RuntimeError("program is finalized (FILL allocated); no new fields")
        if n is None:
            n = len(labels) if kind == "onehot" else 2 * len(labels)
        f = Field(name, kind, tuple(labels), tuple(range(self.ndims, self.ndims + n)), stage, optional)
        self.ndims += n
        self.fields.append(f)
        return f

    def input_onehot(self, name, labels, optional=False):
        return self._alloc(name, "onehot", labels, 0, optional=optional)

    def input_dualrail(self, name, nbits, optional=False):
        return self._alloc(name, "dualrail", range(nbits), 0, optional=optional)

    @staticmethod
    def mlp_layer(inputs, min_layer=0):
        """Earliest layer (not before min_layer) whose MLP sees every input field."""
        m = max(f.stage for f in inputs)
        return max(min_layer, math.ceil((m - 1) / 2))

    def mlp_out(self, name, kind, labels, inputs, min_layer=0):
        layer = self.mlp_layer(inputs, min_layer)
        return layer, self._alloc(name, kind, labels, 2 * layer + 2)

    def add_unit(self, layer, gate, up, down):
        self.units[layer].append(Unit(dict(gate), dict(up), dict(down)))

    def const_unit_down(self, layer):
        """Down-projection dict of the layer's constant unit, which outputs exactly 1."""
        if layer not in self._const_unit:
            u = Unit({self.one.dims[0]: G}, {self.one.dims[0]: 1.0 / G}, {})
            self.units[layer].append(u)
            self._const_unit[layer] = u.down
        return self._const_unit[layer]

    # ---- tokens -----------------------------------------------------------
    def set_vocab(self, vocab, embed_fn):
        """embed_fn(token) -> {Field: value} for stage-0 fields. ONE and FILL are added here."""
        self.vocab = list(vocab)
        self.embed_fn = embed_fn

    def set_emit(self, emit_fn):
        """emit_fn(token) -> [(Field, value)] that the residual must hold to emit it, or None."""
        self.emit_fn = emit_fn

    # ---- compile-time facts --------------------------------------------------
    @property
    def n_layers(self):
        used = [l for l, us in self.units.items() if us]
        return max(used + [0]) + 1

    def embed_matrix(self):
        """Embedding rows (vocab x ndims) with FILL; validates fixed counts."""
        fill = self.fill
        rows = np.zeros((len(self.vocab), self.ndims))
        norms = []
        for i, tok in enumerate(self.vocab):
            assign = self.embed_fn(tok)
            for f in self.fields:
                if f.stage != 0 or f.kind not in ("onehot", "dualrail"):
                    continue
                if f in assign:
                    rows[i, f.encode(assign[f])] = 1.0
                elif not f.optional:
                    raise ValueError(f"token {tok}: required field {f.name} unset")
            for h in self.heads:  # a write must store a complete pattern
                if h.write_slot in assign and any(f not in assign for f in h.write_values):
                    raise ValueError(f"token {tok}: head {h.name} writes without a value")
            rows[i, self.one.dims[0]] = 1.0
            norms.append(rows[i].sum())
        c = max(norms)
        for i, n in enumerate(norms):
            rows[i, fill.dims[0]] = math.sqrt(c - n)
        self.embed_norm2 = c
        return rows

    @property
    def fill(self):
        if not hasattr(self, "_fill"):
            self._fill = Field("FILL", "fill", (), (self.ndims,), 0)
            self.ndims += 1
        return self._fill

    def norm2_at(self, stage):
        """Squared residual norm at `stage`; identical on every token by construction."""
        return self.embed_norm2 + sum(f.count for f in self.fields if 0 < f.stage <= stage)

    def rms_scale(self, stage):
        """The factor RMSNorm multiplies the residual by at `stage`."""
        d = self.ndims
        return 1.0 / math.sqrt(self.norm2_at(stage) / d + RMS_EPS)

    # ---- reference --------------------------------------------------------
    def reference(self, tokens):
        """Ideal residual after the last layer for each token."""
        ref = self.reference_stepper()
        return [ref(tok) for tok in tokens]

    def reference_stepper(self):
        """Function token -> ideal residual, carrying memory state across calls."""
        if not hasattr(self, "_emb"):
            self._emb = self.embed_matrix()
            self._index = {t: i for i, t in enumerate(self.vocab)}
        state = {h.name: [None] * h.n_slots for h in self.heads}

        def step(tok):
            x = self._emb[self._index[tok]].copy()
            for kind, fn in self.refs:
                fn(x, state) if kind == "mem" else fn(x)
            return x

        return step

    def reference_next(self, x):
        """Token whose emit components all hold in ideal residual x."""
        hits = [t for t in self.vocab if self._emits(t, x)]
        assert len(hits) == 1, f"{len(hits)} tokens match"
        return hits[0]

    def _emits(self, tok, x):
        comps = self.emit_fn(tok)
        return comps is not None and all(x[d] == 1.0 for f, v in comps for d in f.encode(v))
