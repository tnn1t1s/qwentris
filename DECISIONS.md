# Decisions

Each entry gives the choice, the reason, and the SPEC section it amends or fills in.

## D1. The ISA interpreter has the model's information limit (§3.4, P0)

`isa.next_token(tok, mem)` takes only the current token and the memory map after
that token's write. It has no trace history and no host state, and it raises
`UnwrittenRead` if any slot is unwritten. The compiled model can see nothing else,
so an interpreter that saw more could pass P0 with a protocol the model cannot run.
D2 through D5 are protocol gaps this constraint exposed.

## D2. BOS writes the whole initial memory; the WROW boot loop is removed (§3.1, §3.2, §3.4)

The SPEC boot emits `WROW(r, 0)` for r = 19…0. During those 20 tokens most
board slots and every register are unread-before-written, so the reads return
zero and the residual norm differs from every later token, which breaks §3.2.

Replica Bi is only ever queried at slot i, so one token can write slot i of every
replica Bi at once (each head has its own k projection). BOS therefore writes all
20 empty rows, PIECE (a placeholder spawn value), PHASE = SPAWN, T = −1 and
S = −1. The boot trace is `BOS → NEED_PIECE`.

The same observation means a `WROW(r, …)` only needs to write replica Br. Writing
the other 19 replicas, as §3.1 says, stores values at slots nobody reads. The
compiler writes only Br.

## D3. SETP carries a lock bit (§3.3, §3.4)

When an action is legal but gravity is blocked, the piece locks at the
post-action position. In the SPEC, PLAY emits `PHASE(LOCK)` directly, so PIECE
still holds the pre-action position and the LOCK rows are wrong. Emitting
`SETP(post-action)` first does not fix this by itself, because the token after a
SETP in PLAY phase would then be either `TICK_END` or `PHASE(LOCK)` depending on
information the model no longer has.

`SETP` gains a field `lock ∈ {0, 1}` (it is not stored in PIECE). On a lock tick
the model emits `SETP(post-action, lock=1)`, then `PHASE(LOCK)`. On a move tick
it emits `SETP(post-gravity, lock=0)`, then `TICK_END`. The SETP family doubles
to 14 784 tokens. `test_lock_after_legal_action_uses_post_action_position`
covers this case.

## D4. PHASE(LOCK) and PHASE(CLEAR) reset T (§3.1, §3.4)

LOCK needs "next piece row after the last one written", and CLEAR needs
"t = T − 1, starting at 19". In both cases T on entry holds a stale row from
earlier writes. `PHASE(LOCK)` now writes T ← −1 and `PHASE(CLEAR)` writes
T ← 20, so the T register range is −1…20 (22 values).

## D5. LINES(k) is emitted at the start of CLEAR (§3.4)

At the end of compaction no register holds k: after compaction the board no
longer shows which rows were full. At `PHASE(CLEAR)` the post-lock board is
intact, so the model counts full rows there and emits `LINES(k)`. The CLEAR
sequence is `PHASE(CLEAR) → LINES(k) → SRC(s) → WROW(19, …) → SRC → WROW(18, …)
… → WROW(0, …) → PHASE(SPAWN) → NEED_PIECE`.

## D6. O piece cells are box columns 0–1 (§2)

With SPEC spawn x = 4, this puts O in board columns 4–5, the guideline spawn
columns. JLSTZ use a 3×3 rotation box and I a 4×4 box, rotated clockwise as
(r, c) → (c, n − 1 − r), which reproduces the SRS bitmaps.

## D7. P0 uses a second, greedy game set (§4 P0)

10 000 random-action games produced 61 lines in total and no multi-line clears,
so random play alone barely exercises CLEAR. The P0 test also runs 500 games of
100 pieces with a greedy placement player (`qwentris/ref/players.py`), which gives
18 348 lines and 1 348 multi-line clears. P2 should use both sets.

## D8. transformers pinned to 5.22.1 (§1)

The §1 facts were checked by reading `modeling_qwen3_5.py` and by
`tests/test_arch_facts.py`. One detail the SPEC omits: `a` passes through
`softplus(a + dt_bias)`, where `dt_bias` is a per-head parameter initialised to 1.
This does not matter, because with `A_log = −40` the decay is 1.0 for any
softplus value up to about 1e10.

The pure-PyTorch delta rule and conv are fallbacks behind
`use_kernel_func_from_hub_with_fallback`. If `fla` or `causal_conv1d` is
installed, different kernels run. The environment must not include them.

## D9. §7.5 becomes a difficulty ladder; `qwentris-rotated` tests wiring only (§7.5, P4)

Rotating the residual stream changes only the basis of the residual stream that
connects the modules. SiLU acts on each coordinate of q, k, v and of the MLP
hidden units, which fixes their basis, so per-head keys, values, state and MLP
units stay readable after rotation. `qwentris-rotated` therefore tests whether
the subject can trace wiring between modules, not whether it depends on sparsity.

The headline becomes READ and FIX scores across four variants of increasing
difficulty:

1. `qwentris-clean`
2. `qwentris-rotated`: residual rotation plus MLP permutation (P4 as written)
3. `qwentris-dummy`: rotated plus dense dummy circuits whose outputs do not
   reach the logits; the function stays exact
4. `qwentris-distilled`: same architecture trained to imitate `isa.py` traces

The point where scores drop says which kind of structure the subject relies on.
Variant 4 differs in kind (see open questions below) and is the only part of the
project that needs training; it runs on a GPU, not a laptop.

Open questions for variant 4, to settle before it is built:
- Behaviour can be graded with the differential test, but the distilled model
  will not be exact. Its baseline pass rate has to be measured and FIX scores
  read relative to it.
- READ questions 3–5 and FIX grading (weights changed versus the minimal
  ground-truth diff, correct component) need ground truth about internals. A
  trained model has none, so for variant 4 those items need a different grader,
  or only behaviour-level items are scored.
- Whether the distilled model shows superposition is a hypothesis to measure,
  not a property of distillation.

## D10. MLP primitives are exact gated copies and integer steps (§4 P1)

Every residual dim except FILL is exactly 0 or 1. Two fp32 identities,
`silu(30)/30 == 1.0` and `silu(0) == 0`, make `silu(G·s)·(x/G) = s AND x`
exact, so mux, AND and each LUT row cost one unit. A threshold on an integer sum
uses two units, `(silu(G(s−t+1)) − silu(G(s−t)))/G = [s ≥ t]`, plus a shared
constant unit for the off-rail. NOT and bijective relabels are views over
existing dims and cost nothing. The SPEC's "±30 selector" mux is the one-unit
gated copy.

RMSNorm weights stay 0. The compiler tracks the residual norm at every stage
(it is constant by §3.2) and folds the resulting scale into the next projection.

## D11. Runner passes position_ids and a dict attention mask (§1, P2)

Observed on transformers 5.22.1: a model whose layers are all `linear_attention`
raises in `DynamicCache.get_seq_length()` on a cached forward pass, unless
`position_ids` is given and `attention_mask` is a dict keyed by layer type
(`{"linear_attention": None}`), which skips building the full-attention mask.
`qwentris.compiler.emit_hf.Runner` does both. The model still loads with
`AutoModelForCausalLM` and no custom code. `generate()` raises the same
`ValueError` (observed on a one-call CPU probe of `models/qwentris`), so the
model runs only under the host loop.

## D12. Board heads hold one slot each; a display head holds the whole board (§3.1, P3)

Following D2, head Bi stores only row i, at slot 0. A 21st board head, DISPLAY,
has 20 slots and is written by every WROW at slot r, and its out_proj is 0. The
model never reads it, but its `[k_dim × v_dim]` state matrix is the board, which
gives P3 its heatmap and P2 a single place to decode the board from. The P1
memory test checks the DISPLAY decode.


## D13. Qwentris model size and circuit (§4 P2)

The compiler picks the sizes; they are recorded here, not tuned. `compile_qwentris()`
(qwentris/build.py) gives hidden size 1681, 6 layers, intermediate size 1860,
25 DeltaNet heads (k_dim 20, v_dim 45), vocabulary 35 309: 219.7M parameters,
838 MB as fp32 safetensors. Most of it is the embedding and LM head over the
vocabulary.

- Layer count is the circuit depth under ASAP scheduling. The longest chain is
  the ACT path: piece reads → candidate rotation/x → candidate mask → mask AND
  board window → collision OR → position and lock → output mux. Every other
  path fits within those six MLPs.
- Heads: B0..B19 (one row each, D12), DISPLAY, PIECE (type, rot, x, y onehots in
  one value), PHASE, T, S. All live in layer 0; mixers in layers 1–5 have
  out_proj = 0.
- Intermediate size is the largest per-layer unit count (layer 1, 1860 units).
  The board windows are placed in layer 1 rather than layer 0 (`min_layer`) to
  even out the two layers.
- DISPLAY is never cleared. BOS writes every B row but cannot write 20 DISPLAY
  slots in one token (one key per head per token). An unwritten DISPLAY slot
  holds zero state, and zero decodes as an empty row, so a fresh cache shows
  the empty board. Each game starts a fresh cache.
- The model never sees an unreachable input pattern in the differential test.
  Where a table must still be total, the unreachable entries are listed in
  build.py comments (WROW outside LOCK/CLEAR; no-lock at y = 19).

## D14. The differential test feeds the isa trace (§4 P2)

The model is fed the isa trace and every token it must emit is checked against
argmax. If all of them match, the model's own greedy rollout is that trace, so
this is equal to a free run and lets B games share one batched forward. Padding
after a game ends is never checked. The test runs on an NVIDIA GB10 GPU (fp32 with
TF32 off for matmul and cuDNN). On the laptop the stock depthwise conv1d runs as
a per-channel loop at about 150 ms per token, so laptop tests use short games.

## D15. Visualisation is for the demo only (§4 P3)

The visualisation is for the demo only. P3 delivers the MP4
(board, DISPLAY heatmap, register strips, token trace) and skips the
layer × unit activation raster.
