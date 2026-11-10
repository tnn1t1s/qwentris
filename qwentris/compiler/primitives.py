"""Circuit primitives. Each one allocates its output field, adds MLP units or a
memory head, and registers a reference function.

MLP units rely on two fp32 identities (checked in tests/test_primitives.py):
silu(G) / G == 1.0 and silu(0) == 0. With every residual value exactly 0 or 1:

- gated copy: silu(G * s) * (x / G) == s AND x, so one unit gives mux, AND, LUT rows;
- step on an integer sum s: (silu(G(s - t + 1)) - silu(G(s - t))) / G == [s >= t],
  so two units give any threshold function (and_k, or_k, majority, ...).

A term is (field, dim, integer coefficient). `lit` and `is_` build them.
"""

from itertools import product

from qwentris.compiler.graph import G, Head
from qwentris.compiler.layout import Field


def lit(f, bit=0, on=True):
    return (f, f.lit(bit, on), 1)


def is_(f, label):
    return (f, f.at(label), 1)


def thresholds(prog, name, specs, min_layer=0):
    """Dualrail field with bit i = [sum(terms_i) >= t_i], for specs [(terms_i, t_i)]."""
    inputs = [f for terms, _ in specs for f, _, _ in terms]
    for terms, t in specs:
        assert all(float(c).is_integer() for _, _, c in terms) and float(t).is_integer()
    layer, out = prog.mlp_out(name, "dualrail", range(len(specs)), inputs, min_layer)
    one = prog.one.dims[0]
    const = prog.const_unit_down(layer)
    for i, (terms, t) in enumerate(specs):
        on, off = out.lit(i, True), out.lit(i, False)
        gate = {}
        for _, d, c in terms:
            gate[d] = gate.get(d, 0.0) + G * c
        hi = dict(gate)
        hi[one] = hi.get(one, 0.0) + G * (1 - t)
        lo = dict(gate)
        lo[one] = lo.get(one, 0.0) - G * t
        prog.add_unit(layer, hi, {one: 1.0 / G}, {on: 1.0, off: -1.0})
        prog.add_unit(layer, lo, {one: 1.0 / G}, {on: -1.0, off: 1.0})
        const[off] = const.get(off, 0.0) + 1.0

    def ref(x):
        for i, (terms, t) in enumerate(specs):
            b = sum(c * x[d] for _, d, c in terms) >= t
            x[out.lit(i, True)], x[out.lit(i, False)] = float(b), float(not b)

    prog.refs.append(("mlp", ref))
    return out


def threshold(prog, name, terms, t):
    return thresholds(prog, name, [(terms, t)])


def and_(prog, name, terms):
    return threshold(prog, name, terms, len(terms))


def or_(prog, name, terms):
    return threshold(prog, name, terms, 1)


def lut(prog, name, inputs, fn, kind, labels, min_layer=0):
    """Table lookup over the product of onehot inputs. fn(*labels) -> output value.

    kind "onehot": output labels `labels`; kind "dualrail": `labels` = range(nbits).
    One unit per input combination. fn must be total over the product.
    """
    assert all(f.kind == "onehot" for f in inputs)
    layer, out = prog.mlp_out(name, kind, labels, inputs, min_layer)
    one = prog.one.dims[0]
    n = len(inputs)
    for combo in product(*(f.labels for f in inputs)):
        gate = {f.at(u): G for f, u in zip(inputs, combo)}
        gate[one] = G * (1 - n)
        prog.add_unit(layer, gate, {one: 1.0 / G}, {d: 1.0 for d in out.encode(fn(*combo))})

    def ref(x):
        v = fn(*(f.decode(x) for f in inputs))
        x[list(out.dims)] = 0.0
        x[out.encode(v)] = 1.0

    prog.refs.append(("mlp", ref))
    return out


def linear_map(prog, name, f, fn, labels):
    """Onehot -> onehot function (e.g. x+1 with an out-of-range label). Use relabel for bijections."""
    return lut(prog, name, [f], fn, "onehot", labels)


def mux(prog, name, sel, inputs, like, min_layer=0):
    """out = inputs[sel]. inputs: {sel label: Field or constant value}; `like` gives the shape.

    Dualrail: one unit per selected on-rail; the off-rail is 1 - on (constant unit).
    Onehot: sources are aligned by label; a source label missing from the output has
    no unit, and the reference raises if it is ever selected.
    """
    assert sel.kind == "onehot" and set(inputs) == set(sel.labels), name
    srcs = [v for v in inputs.values() if isinstance(v, Field)]
    layer, out = prog.mlp_out(name, like.kind, like.labels, [sel] + srcs, min_layer)
    one = prog.one.dims[0]
    if out.kind == "dualrail":
        const = prog.const_unit_down(layer)
        for b in range(out.nbits):
            const[out.lit(b, False)] = const.get(out.lit(b, False), 0.0) + 1.0
    for s, src in inputs.items():
        g = {sel.at(s): G}
        if isinstance(src, Field):
            assert src.kind == out.kind, (name, src.name)
            if out.kind == "dualrail":
                assert src.nbits == out.nbits, (name, src.name)
                for b in range(out.nbits):
                    prog.add_unit(layer, g, {src.lit(b, True): 1.0 / G},
                                  {out.lit(b, True): 1.0, out.lit(b, False): -1.0})
            else:
                for u in src.labels:
                    if u in out.labels:
                        prog.add_unit(layer, g, {src.at(u): 1.0 / G}, {out.at(u): 1.0})
        elif out.kind == "dualrail":
            on = [b for b in range(out.nbits) if src >> b & 1]
            if on:
                down = {}
                for b in on:
                    down[out.lit(b, True)], down[out.lit(b, False)] = 1.0, -1.0
                prog.add_unit(layer, g, {one: 1.0 / G}, down)
        else:
            prog.add_unit(layer, g, {one: 1.0 / G}, {out.at(src): 1.0})

    def ref(x):
        src = inputs[sel.decode(x)]
        v = src.decode(x) if isinstance(src, Field) else src
        if out.kind == "onehot" and v not in out.labels:
            raise RuntimeError(f"{name}: selected value {v!r} has no output label")
        x[list(out.dims)] = 0.0
        x[out.encode(v)] = 1.0

    prog.refs.append(("mlp", ref))
    return out


def and2(prog, name, pairs, min_layer=0):
    """Dualrail field with bit i = a_i AND b_i, for pairs [(term a_i, term b_i)].

    One gated-copy unit per bit: silu(G * a) * (b / G); the off-rail is 1 - on.
    """
    inputs = [f for pair in pairs for f, _, _ in pair]
    layer, out = prog.mlp_out(name, "dualrail", range(len(pairs)), inputs, min_layer)
    const = prog.const_unit_down(layer)
    for i, ((_, a, ca), (_, b, cb)) in enumerate(pairs):
        assert ca == 1 and cb == 1
        prog.add_unit(layer, {a: G}, {b: 1.0 / G}, {out.lit(i, True): 1.0, out.lit(i, False): -1.0})
        const[out.lit(i, False)] = const.get(out.lit(i, False), 0.0) + 1.0

    def ref(x):
        for i, ((_, a, _), (_, b, _)) in enumerate(pairs):
            v = x[a] == 1.0 and x[b] == 1.0
            x[out.lit(i, True)], x[out.lit(i, False)] = float(v), float(not v)

    prog.refs.append(("mlp", ref))
    return out


def _step_units(prog, layer, terms, t, down):
    """Two units adding [sum(terms) >= t] * down."""
    one = prog.one.dims[0]
    gate = {}
    for _, d, c in terms:
        gate[d] = gate.get(d, 0.0) + G * c
    hi, lo = dict(gate), dict(gate)
    hi[one] = hi.get(one, 0.0) + G * (1 - t)
    lo[one] = lo.get(one, 0.0) - G * t
    prog.add_unit(layer, hi, {one: 1.0 / G}, dict(down))
    prog.add_unit(layer, lo, {one: 1.0 / G}, {d: -w for d, w in down.items()})


def first_match(prog, name, labels, rest, specs, any_name=None, min_layer=0):
    """Onehot over `labels`: label u when one spec (u, terms, t) has sum(terms) >= t,
    else `rest`. At most one spec may fire (the reference asserts it). Optionally
    also returns a dualrail bit `any_name` that is 1 when a spec fired.
    """
    assert rest in labels
    inputs = [f for _, terms, _ in specs for f, _, _ in terms]
    layer, out = prog.mlp_out(name, "onehot", labels, inputs, min_layer)
    anyf = prog._alloc(any_name, "dualrail", range(1), 2 * layer + 2) if any_name else None
    const = prog.const_unit_down(layer)
    const[out.at(rest)] = const.get(out.at(rest), 0.0) + 1.0
    if anyf:
        const[anyf.lit(0, False)] = const.get(anyf.lit(0, False), 0.0) + 1.0
    for u, terms, t in specs:
        assert u != rest
        down = {out.at(u): 1.0, out.at(rest): -1.0}
        if anyf:
            down[anyf.lit(0, True)], down[anyf.lit(0, False)] = 1.0, -1.0
        _step_units(prog, layer, terms, t, down)

    def ref(x):
        hits = [u for u, terms, t in specs if sum(c * x[d] for _, d, c in terms) >= t]
        if len(hits) > 1:
            raise RuntimeError(f"{name}: {len(hits)} specs fire ({hits})")
        v = hits[0] if hits else rest
        x[list(out.dims)] = 0.0
        x[out.at(v)] = 1.0
        if anyf:
            x[anyf.lit(0, True)], x[anyf.lit(0, False)] = float(bool(hits)), float(not hits)

    prog.refs.append(("mlp", ref))
    return (out, anyf) if anyf else out


def count_onehot(prog, name, terms, kmax, min_layer=0):
    """Onehot over 0..kmax of min(sum(terms), kmax). Label k = [s >= k] - [s >= k+1]."""
    inputs = [f for f, _, _ in terms]
    labels = tuple(range(kmax + 1))
    layer, out = prog.mlp_out(name, "onehot", labels, inputs, min_layer)
    const = prog.const_unit_down(layer)
    const[out.at(0)] = const.get(out.at(0), 0.0) + 1.0
    for k in range(1, kmax + 1):
        _step_units(prog, layer, terms, k, {out.at(k): 1.0, out.at(k - 1): -1.0})

    def ref(x):
        v = min(int(sum(c * x[d] for _, d, c in terms)), kmax)
        x[list(out.dims)] = 0.0
        x[out.at(v)] = 1.0

    prog.refs.append(("mlp", ref))
    return out


def rules(prog, name, outs, table, min_layer=0):
    """Rule table. outs: [(name, kind, labels)]. table: [(conds, values)], where conds
    is a list of (field, dim, want) with want 1 or 0 and values is {out name: value}.

    One unit per rule: it fires when every cond holds and writes its values. Exactly
    one rule must fire on every token (the reference asserts it); an output a rule
    does not name gets no write from that rule, so every rule names every output.
    """
    inputs = [f for conds, _ in table for f, _, _ in conds]
    layer = prog.mlp_layer(inputs, min_layer)
    fields = {n: prog._alloc(n, k, l, 2 * layer + 2) for n, k, l in outs}
    one = prog.one.dims[0]
    for conds, values in table:
        assert set(values) == set(fields), (name, values)
        npos = sum(1 for _, _, w in conds if w == 1)
        gate = {one: G * (1 - npos)}
        for _, d, w in conds:
            gate[d] = gate.get(d, 0.0) + (G if w == 1 else -G)
        down = {d: 1.0 for n, v in values.items() for d in fields[n].encode(v)}
        prog.add_unit(layer, gate, {one: 1.0 / G}, down)

    def ref(x):
        hits = [values for conds, values in table if all(x[d] == w for _, d, w in conds)]
        if len(hits) != 1:
            raise RuntimeError(f"{name}: {len(hits)} rules fire")
        for n, f in fields.items():
            x[list(f.dims)] = 0.0
            x[f.encode(hits[0][n])] = 1.0

    prog.refs.append(("mlp", ref))
    return fields


def memory(prog, name, n_slots, write_slot, write_values, read_slot=0, read=True):
    """A layer-0 memory head. Writes the concatenated dims of write_values (a Field or a
    list of Fields) at the slot write_slot names (no write when write_slot is absent from
    the token); reads slot read_slot after the write.

    Returns the read field(s) at stage 1, shaped like write_values (a single Field when
    one was given), or None if read=False."""
    single = isinstance(write_values, Field)
    values = [write_values] if single else list(write_values)
    assert write_slot.stage == 0 and all(f.stage == 0 for f in values), "writes come from the token only"
    assert write_slot.kind == "onehot" and write_slot.labels == tuple(range(n_slots))
    outs = None
    if read:
        outs = [prog._alloc(f"{name}.read.{f.name}", f.kind, f.labels, 1) for f in values]
    head = Head(name, n_slots, write_slot, values, read_slot, outs)
    prog.heads.append(head)
    vdims = head.value_dims

    def ref(x, state):
        slots = state[name]
        on = [j for j in range(n_slots) if x[write_slot.dims[j]] == 1.0]
        if on:
            slots[on[0]] = x[vdims].copy()
        if outs is not None:
            if slots[read_slot] is None:
                raise RuntimeError(f"{name}: read of unwritten slot {read_slot}")
            x[head.out_dims] = slots[read_slot]

    prog.refs.append(("mem", ref))
    if outs is None:
        return None
    return outs[0] if single else outs
