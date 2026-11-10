"""Residual-stream layout: named fields of 0/1 dimensions.

Every residual dimension except FILL holds exactly 0 or 1 on every token. A field
is one of:

- onehot: one dim per label, exactly one dim is 1 (or none, for optional
  embedding fields);
- dualrail: n bits, two dims per bit (on-rail, off-rail), exactly one rail is 1;
- one: the constant-1 dimension that stands in for every bias;
- fill: the embedding padding dim that makes every embedding row the same norm.

Fields are immutable. `not_`, `bits` and `relabel` return views over existing
dims and cost no compute.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    labels: tuple
    dims: tuple
    stage: int
    optional: bool = False

    @property
    def count(self):
        """Number of dims that are 1 on every token (the field's squared norm)."""
        if self.kind == "dualrail":
            return len(self.dims) // 2
        if self.kind in ("onehot", "one"):
            return 1
        raise ValueError(f"{self.name}: no fixed count for kind {self.kind}")

    @property
    def nbits(self):
        assert self.kind == "dualrail"
        return len(self.dims) // 2

    def at(self, label):
        assert self.kind == "onehot", self.name
        return self.dims[self.labels.index(label)]

    def lit(self, bit=0, on=True):
        """Dim that is 1 exactly when bit `bit` equals `on`."""
        assert self.kind == "dualrail", self.name
        return self.dims[2 * bit + (0 if on else 1)]

    def encode(self, value):
        """Dims that are 1 when the field holds `value`."""
        if self.kind == "onehot":
            return [self.at(value)]
        if self.kind == "dualrail":
            return [self.lit(b, bool(value >> b & 1)) for b in range(self.nbits)]
        if self.kind == "one":
            return list(self.dims)
        raise ValueError(self.kind)

    def decode(self, vec):
        """Field value from a residual vector, by argmax (onehot) or rail comparison."""
        v = np.asarray(vec)[list(self.dims)]
        if self.kind == "onehot":
            return self.labels[int(np.argmax(v))]
        if self.kind == "dualrail":
            return sum(1 << b for b in range(self.nbits) if v[2 * b] > v[2 * b + 1])
        raise ValueError(self.kind)


def not_(f):
    """Logical NOT of a dualrail field: swap the rails."""
    assert f.kind == "dualrail"
    d = f.dims
    swapped = tuple(x for i in range(0, len(d), 2) for x in (d[i + 1], d[i]))
    return replace(f, name=f"not({f.name})", dims=swapped)


def bits(f, idx):
    """Dualrail view of the selected bits of a dualrail field, in the given order."""
    assert f.kind == "dualrail"
    dims = tuple(x for b in idx for x in (f.lit(b, True), f.lit(b, False)))
    return replace(f, name=f"{f.name}{list(idx)}", labels=tuple(range(len(idx))), dims=dims)


def relabel(f, fn):
    """Onehot view with label u renamed fn(u). fn must be injective."""
    assert f.kind == "onehot"
    labels = tuple(fn(u) for u in f.labels)
    assert len(set(labels)) == len(labels), "relabel needs an injective map; use lut otherwise"
    return replace(f, name=f"relabel({f.name})", labels=labels)


def as_onehot(f, bit=0):
    """Two-label onehot view (False, True) of one bit of a dualrail field."""
    assert f.kind == "dualrail"
    return replace(f, name=f"{f.name}[{bit}]?", kind="onehot", labels=(False, True),
                   dims=(f.lit(bit, False), f.lit(bit, True)))
