"""compile_deltris(): the Deltris protocol (deltris/ref/isa.py) as a Program.

Memory (layer-0 heads, every one read after its write):
  R0..R19  one slot each, a 10-bit dualrail board row (D2, D12);
  DISPLAY  20 slots, written by every WROW at slot r, never read (D12);
  PIECE    type, rot, x, y onehots;  PHASE, T, S onehots.

The MLPs compute the next token from the current token and those reads. Layer
numbers below are where ASAP scheduling puts each op (see Program.mlp_layer).
"""

from deltris.compiler import primitives as P
from deltris.compiler.graph import Program
from deltris.compiler.layout import as_onehot, bits, not_, relabel
from deltris.ref import isa
from deltris.ref.tetris import (
    ACTIONS,
    FULL_ROW,
    HEIGHT,
    PIECES,
    SHAPES,
    WIDTH,
    X_RANGE,
    Y_RANGE,
    piece_row_mask,
    spawn_position,
)

KINDS = ("BOS", "ACT", "PIECE", "WROW", "SETP0", "SETP1", "SRC",
         "PH_SPAWN", "PH_PLAY", "PH_LOCK", "PH_CLEAR", "PH_OVER", "LINES", "STOP")
OUT_KINDS = ("NEED_PIECE", "TICK_END", "GAME_OVER", "SETP", "PHASE", "WROW", "SRC", "LINES")
BOX = 4  # invariant: every piece fits a 4x4 box; mask rows k = 0..3 below the box top
WINDOWS = BOX + 1  # rows y..y+4 cover the piece at y and one row down
OOB = "OOB"  # cand_x label for x-1 = -3 or x+1 = 10: always illegal (see _box_mask)
NA = "NA"
MAX_FULL = 5  # count_onehot cap; at most 4 rows are full at once (one piece spans 4 rows)


def token_kind(tok):
    k = tok[0]
    if k == "SETP":
        return f"SETP{tok[5]}"
    if k == "PHASE":
        return f"PH_{tok[1]}"
    if k in ("TICK_END", "NEED_PIECE", "GAME_OVER"):
        return "STOP"
    return k


def _box_mask(p, rot, x):
    """(mask per box row k as a 10-bit row, any cell outside columns 0..9)."""
    rows, hoob = [0] * BOX, False
    for r, c in SHAPES[p][rot]:
        col = x + c
        if 0 <= col < WIDTH:
            rows[r] |= 1 << col
        else:
            hoob = True
    return rows, hoob


def _mask_bits(p, rot, x):
    """41-bit value: bits 10k+c = box row k column c; bit 40 = horizontal out of bounds."""
    if x == OOB:
        return 1 << (BOX * WIDTH)
    rows, hoob = _box_mask(p, rot, x)
    v = sum(row << (WIDTH * k) for k, row in enumerate(rows))
    return v | (int(hoob) << (BOX * WIDTH))


def _occ_bits(p, rot, y):
    """20-bit value: board rows the piece covers, rows outside the board dropped."""
    v = 0
    for r, _ in SHAPES[p][rot]:
        if 0 <= y + r < HEIGHT:
            v |= 1 << (y + r)
    return v


def compile_deltris():
    prog = Program()
    vocab = isa.vocabulary()

    # ---- token fields (stage 0) ------------------------------------------
    kind = prog.input_onehot("kind", KINDS)
    act = prog.input_onehot("act", ACTIONS)  # NOOP on non-ACT tokens
    ptype = prog.input_onehot("ptype", PIECES)  # spawned type on PIECE tokens, else I
    spawn_x = prog.input_onehot("spawn_x", X_RANGE)  # spawn x on PIECE tokens, else 3
    sm = [prog.input_dualrail(f"spawn_mask{k}", WIDTH) for k in range(2)]  # rows 0, 1
    rowval = prog.input_dualrail("rowval", WIDTH)  # WROW bits; 0 elsewhere (BOS writes it)
    wr = [prog.input_onehot(f"wr{r}", (0,), optional=True) for r in range(HEIGHT)]
    disp_slot = prog.input_onehot("disp_slot", tuple(range(HEIGHT)), optional=True)
    wp = prog.input_onehot("wp", (0,), optional=True)
    pv = [prog.input_onehot("pv_type", PIECES, optional=True),
          prog.input_onehot("pv_rot", tuple(range(4)), optional=True),
          prog.input_onehot("pv_x", X_RANGE, optional=True),
          prog.input_onehot("pv_y", Y_RANGE, optional=True)]
    wph = prog.input_onehot("wph", (0,), optional=True)
    phv = prog.input_onehot("phv", isa.PHASES, optional=True)
    wt = prog.input_onehot("wt", (0,), optional=True)
    tv = prog.input_onehot("tv", isa.T_RANGE, optional=True)
    ws = prog.input_onehot("ws", (0,), optional=True)
    sv = prog.input_onehot("sv", isa.S_RANGE, optional=True)

    # ---- memory (layer-0 mixer, reads at stage 1) -------------------------
    R = [P.memory(prog, f"B{r}", 1, wr[r], rowval) for r in range(HEIGHT)]
    P.memory(prog, "DISPLAY", HEIGHT, disp_slot, rowval, read=False)
    Pt, Pr, Px, Py = P.memory(prog, "PIECE", 1, wp, pv)
    PH = P.memory(prog, "PHASE", 1, wph, phv)
    T = P.memory(prog, "T", 1, wt, tv)
    S = P.memory(prog, "S", 1, ws, sv)

    def on(f, b=0):
        return P.lit(f, b, True)

    def off(f, b=0):
        return P.lit(f, b, False)

    # ---- layer 0 ---------------------------------------------------------
    full = P.thresholds(prog, "full", [([on(R[r], b) for b in range(WIDTH)], WIDTH) for r in range(HEIGHT)])
    spawn_and = P.and2(prog, "spawn_and", [(on(R[k], b), on(sm[k], b)) for k in range(2) for b in range(WIDTH)])
    rot_labels = tuple(range(4))

    def act_rot(a, r):
        return (r + 1) % 4 if a == "ROT_CW" else r

    def act_x(a, x):
        nx = x + {"LEFT": -1, "RIGHT": 1}.get(a, 0)
        return nx if nx in X_RANGE else OOB

    cand_rot = P.lut(prog, "cand_rot", [act, Pr], act_rot, "onehot", rot_labels)
    cand_x = P.lut(prog, "cand_x", [act, Px], act_x, "onehot", tuple(X_RANGE) + (OOB,))
    M = P.lut(prog, "M", [Pt, Pr, Px], _mask_bits, "dualrail", range(BOX * WIDTH + 1))
    occ = P.lut(prog, "occ", [Pt, Pr, Py], _occ_bits, "dualrail", range(HEIGHT))
    t0 = P.threshold(prog, "T0", [P.is_(T, 0)], 1)

    def start_fn(k, s):
        if k == "LINES":
            return HEIGHT - 1
        if k == "WROW":
            return max(s - 1, -1)
        return -1

    start = P.lut(prog, "start", [kind, S], start_fn, "onehot", isa.S_RANGE)

    # ---- layer 1 ---------------------------------------------------------
    def row_or_full(y, k):
        r = y + k
        return R[r] if 0 <= r < HEIGHT else FULL_ROW

    W = [P.mux(prog, f"W{k}", Py, {y: row_or_full(y, k) for y in Y_RANGE}, R[0], min_layer=1)
         for k in range(WINDOWS)]
    Mc = P.lut(prog, "Mc", [Pt, cand_rot, cand_x], _mask_bits, "dualrail", range(BOX * WIDTH + 1))
    coll_spawn = P.or_(prog, "coll_spawn", [on(spawn_and, i) for i in range(2 * WIDTH)])
    nxt_specs = [(0, [on(occ, 0), P.is_(T, -1)], 2)]
    for r in range(1, HEIGHT):
        nxt_specs.append((r, [on(occ, r), P.is_(T, r - 1)], 2))
        nxt_specs.append((r, [on(occ, r), P.is_(T, -1), off(occ, r - 1)], 3))
    nxt, nxt_any = P.first_match(prog, "next", tuple(range(HEIGHT)) + (NA,), NA, nxt_specs, any_name="next_any")
    count = P.count_onehot(prog, "count", [on(full, r) for r in range(HEIGHT)], MAX_FULL)
    any_full = P.or_(prog, "any_full", [on(full, r) for r in range(HEIGHT)])
    res_specs = []
    for s in range(HEIGHT):
        for r in range(s + 1):
            terms = [P.is_(start, s), off(full, r)] + [on(full, q) for q in range(r + 1, s + 1)]
            res_specs.append((r, terms, len(terms)))
    res = P.first_match(prog, "res", isa.S_RANGE, -1, res_specs)

    # ---- layer 2 ---------------------------------------------------------
    def window_ands(mask, shift):
        return [(on(mask, WIDTH * k + b), on(W[k + shift], b)) for k in range(BOX) for b in range(WIDTH)]

    and_c = P.and2(prog, "and_c", window_ands(Mc, 0))  # cand at y
    and_cd = P.and2(prog, "and_cd", window_ands(Mc, 1))  # cand at y+1
    and_pd = P.and2(prog, "and_pd", window_ands(M, 1))  # current piece at y+1
    rowsel = P.mux(prog, "rowsel", nxt, {**{r: R[r] for r in range(HEIGHT)}, NA: 0}, R[0])

    def jj_fn(n, y):
        return n - y if n != NA and 0 <= n - y < BOX else NA

    jj = P.lut(prog, "jj", [nxt, Py], jj_fn, "onehot", tuple(range(BOX)) + (NA,))

    ph = lambda p: P.is_(PH, p)  # noqa: E731
    k_ = lambda k: P.is_(kind, k)  # noqa: E731

    def nphase(p):
        return (PH, PH.at(p), 0)

    def emit(ok, phase="SPAWN", wsel=NA):
        return {"out_kind": ok, "out_phase": phase, "wsel": wsel}

    lock_rules = [
        ([on(nxt_any)], emit("WROW", wsel="LOCK")),
        ([off(nxt_any), on(any_full)], emit("PHASE", "CLEAR")),
        ([off(nxt_any), off(any_full)], emit("PHASE", "SPAWN")),
    ]
    table = [
        ([k_("BOS")], emit("NEED_PIECE")),
        ([k_("PIECE"), on(coll_spawn)], emit("PHASE", "OVER")),
        ([k_("PIECE"), off(coll_spawn)], emit("SETP")),
        ([k_("SETP1")], emit("PHASE", "LOCK")),
        ([k_("SETP0"), ph("SPAWN")], emit("PHASE", "PLAY")),
        ([k_("SETP0"), nphase("SPAWN")], emit("TICK_END")),
        ([k_("ACT")], emit("SETP")),
        ([k_("PH_PLAY")], emit("TICK_END")),
        ([k_("PH_OVER")], emit("GAME_OVER")),
        ([k_("PH_SPAWN")], emit("NEED_PIECE")),
        ([k_("PH_CLEAR")], emit("LINES")),
        ([k_("LINES")], emit("SRC")),
        ([k_("SRC")], emit("WROW", wsel="SRC")),
        ([k_("STOP")], emit("NEED_PIECE")),  # the host ignores the prediction after a stop token
        ([k_("WROW"), ph("CLEAR"), on(t0)], emit("PHASE", "SPAWN")),
        ([k_("WROW"), ph("CLEAR"), off(t0)], emit("SRC")),
    ]
    table += [([k_("PH_LOCK")] + c, v) for c, v in lock_rules]
    table += [([k_("WROW"), ph("LOCK")] + c, v) for c, v in lock_rules]
    # WROW outside LOCK and CLEAR is unreachable (only the model emits WROW, in those phases)
    table += [([k_("WROW"), ph(p)], emit("NEED_PIECE")) for p in ("SPAWN", "PLAY", "OVER")]
    out = P.rules(prog, "case", [("out_kind", "onehot", OUT_KINDS), ("out_phase", "onehot", isa.PHASES),
                                 ("wsel", "onehot", ("LOCK", "SRC", NA))], table)
    out_kind, out_phase, wsel = out["out_kind"], out["out_phase"], out["wsel"]

    # ---- layer 3 ---------------------------------------------------------
    hoob = BOX * WIDTH
    cc = P.or_(prog, "cc", [on(and_c, i) for i in range(BOX * WIDTH)] + [on(Mc, hoob)])
    cdc = P.or_(prog, "cdc", [on(and_cd, i) for i in range(BOX * WIDTH)] + [on(Mc, hoob)])
    cdp = P.or_(prog, "cdp", [on(and_pd, i) for i in range(BOX * WIDTH)])
    piecebits = P.mux(prog, "piecebits", jj, {**{k: bits(M, range(WIDTH * k, WIDTH * (k + 1)))
                                                 for k in range(BOX)}, NA: 0}, rowval)

    # ---- layer 4 ---------------------------------------------------------
    cc1, cdc1, cdp1 = as_onehot(cc), as_onehot(cdc), as_onehot(cdp)
    pos_rot = P.mux(prog, "pos_rot", cc1, {False: cand_rot, True: Pr}, Pr)
    pos_x = P.mux(prog, "pos_x", cc1, {False: cand_x, True: Px}, Px)

    def lock_fn(c, dc, dp):
        return int(dp if c else dc)

    def pos_y_fn(c, dc, dp, y):
        # not locking means the piece moves to y+1, which is then legal, so y+1 <= 19;
        # y = 19 with no lock is unreachable and maps to 19
        return y if lock_fn(c, dc, dp) else min(y + 1, Y_RANGE[-1])

    lock = P.lut(prog, "lock", [cc1, cdc1, cdp1], lock_fn, "onehot", (0, 1))
    pos_y = P.lut(prog, "pos_y", [cc1, cdc1, cdp1, Py], pos_y_fn, "onehot", Y_RANGE)
    lockrow = P.thresholds(prog, "lockrow", [([on(rowsel, b), on(piecebits, b)], 1) for b in range(WIDTH)])
    srcbits = P.mux(prog, "srcbits", S, {**{s: R[s] for s in range(HEIGHT)}, -1: 0}, R[0], min_layer=4)

    # ---- layer 5: output fields --------------------------------------------
    def by_kind(name, act_src, piece_src, default, like):
        inputs = {k: default for k in KINDS}
        inputs["ACT"], inputs["PIECE"] = act_src, piece_src
        return P.mux(prog, name, kind, inputs, like, min_layer=5)

    sp_type = by_kind("sp_type", Pt, ptype, "I", Pt)
    sp_rot = by_kind("sp_rot", pos_rot, 0, 0, Pr)
    sp_x = by_kind("sp_x", pos_x, spawn_x, 3, Px)
    sp_y = by_kind("sp_y", pos_y, 0, 0, Py)
    sp_lock = by_kind("sp_lock", lock, 0, 0, lock)
    out_row = P.mux(prog, "out_row", wsel, {"LOCK": nxt, "SRC": relabel(T, lambda t: t - 1), NA: 0},
                    disp_slot, min_layer=5)
    out_bits = P.mux(prog, "out_bits", wsel, {"LOCK": lockrow, "SRC": srcbits, NA: 0}, rowval, min_layer=5)

    # ---- vocabulary ---------------------------------------------------------
    def embed(tok):
        k = token_kind(tok)
        a = {kind: k, act: "NOOP", ptype: "I", spawn_x: 3, sm[0]: 0, sm[1]: 0, rowval: 0}
        if k == "ACT":
            a[act] = tok[1]
        elif k == "PIECE":
            sp = spawn_position(tok[1])
            a[ptype], a[spawn_x] = tok[1], sp[2]
            a[sm[0]], a[sm[1]] = piece_row_mask((*sp[:3], 0), 0), piece_row_mask((*sp[:3], 0), 1)
        elif k == "BOS":
            for r in range(HEIGHT):
                a[wr[r]] = 0
            a[wp] = 0
            a.update(zip(pv, isa.BOOT_PIECE))
            a[wph], a[phv] = 0, "SPAWN"
            a[wt], a[tv] = 0, -1
            a[ws], a[sv] = 0, -1
        elif k == "WROW":
            r, b = tok[1], tok[2]
            a[rowval], a[wr[r]], a[disp_slot] = b, 0, r
            a[wt], a[tv] = 0, r
        elif k.startswith("SETP"):
            a[wp] = 0
            a.update(zip(pv, tok[1:5]))
        elif k == "SRC":
            a[ws], a[sv] = 0, tok[1]
        elif k.startswith("PH_"):
            p = tok[1]
            a[wph], a[phv] = 0, p
            if p == "LOCK":
                a[wt], a[tv] = 0, -1
            elif p == "CLEAR":
                a[wt], a[tv] = 0, HEIGHT
        return a

    def emit_fn(tok):
        k = tok[0]
        if k in ("TICK_END", "NEED_PIECE", "GAME_OVER"):
            return [(out_kind, k)]
        if k == "PHASE":
            return [(out_kind, "PHASE"), (out_phase, tok[1])]
        if k == "SETP":
            return [(out_kind, "SETP"), (sp_type, tok[1]), (sp_rot, tok[2]), (sp_x, tok[3]),
                    (sp_y, tok[4]), (sp_lock, tok[5])]
        if k == "WROW":
            return [(out_kind, "WROW"), (out_row, tok[1]), (out_bits, tok[2])]
        if k == "SRC":
            return [(out_kind, "SRC"), (res, tok[1])]
        if k == "LINES":
            return [(out_kind, "LINES"), (count, tok[1])]
        return None  # host tokens: BOS, ACT, PIECE

    prog.set_vocab(vocab, embed)
    prog.set_emit(emit_fn)

    def decode_next(x):
        """Next token from an ideal or computed residual, read field by field."""
        k = out_kind.decode(x)
        if k == "PHASE":
            return ("PHASE", out_phase.decode(x))
        if k == "SETP":
            return ("SETP", *(f.decode(x) for f in (sp_type, sp_rot, sp_x, sp_y, sp_lock)))
        if k == "WROW":
            return ("WROW", out_row.decode(x), out_bits.decode(x))
        if k == "SRC":
            return ("SRC", res.decode(x))
        if k == "LINES":
            return ("LINES", count.decode(x))
        return (k,)

    prog.decode_next = decode_next
    prog.display_head = [h.name for h in prog.heads].index("DISPLAY")
    return prog


def save(out_dir):
    """Compile and write a stock Qwen3_5ForCausalLM checkpoint (fp32 safetensors)."""
    from deltris.compiler.emit_hf import build, load

    prog = compile_deltris()
    config, sd = build(prog)
    model = load(config, sd)
    model.save_pretrained(out_dir)
    return prog, config


if __name__ == "__main__":
    import sys

    prog, config = save(sys.argv[1])
    print(f"wrote {sys.argv[1]}: hidden {config.hidden_size}, layers {config.num_hidden_layers}, "
          f"intermediate {config.intermediate_size}, heads {config.linear_num_value_heads}, "
          f"vocab {config.vocab_size}")
