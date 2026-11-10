# SPEC: Deltris — Tetris compiled directly into Gated DeltaNet weights

Status: research spec. Audience: an implementation agent plus the project owner.
Sections 1–6 are engineering and should be followed literally. Section 7 is the research protocol.

---

## 0. What we are building and why

A tiny **stock Qwen3.5 architecture** model (loads with `transformers` as `Qwen3_5ForCausalLM`).
**No weight is trained.** A compiler computes every weight, so that:

- The **Gated DeltaNet recurrent state is the Tetris board.** Each board row is one memory slot.
  If you plot the state matrix, you see the game.
- The model plays by **writing its own memory**. Each token the model emits is a memory write
  (a row, a piece register, a phase), and the token is fed back in as the next input. The host only
  supplies player actions and the next piece.
- The model behaves **exactly** like a reference Python Tetris on every token of every game.

**Why this and not an emulator:** Qwenputer (github.com/divagr18/qwenputer) and its forks already
showed that you can compile a CPU into weights with DeltaNet state as RAM. Running a CHIP-8 Tetris ROM on a compiled
CPU would only reproduce that result. Compiling the game's *own* rules as a circuit (no CPU, no ISA
emulation) gives a small, fully legible network whose ground truth we own. That is the basis for
the weight-space-programming experiments in §7 (Takehi et al., "Sparks of Weight-Space Programming",
MIT CSAIL, Nov 2026).

**Non-goals for v1:** wall kicks (SRS), hold, hard drop, scoring beyond a line count, in-model
RNG, grafting into Qwen3.5-4B. These are listed as stretch goals in §8.

---

## 1. Verified architecture facts (from `transformers` `modeling_qwen3_5.py`)

The compiler depends on these facts. Write a test for each one (§4, P1).

**Decoder layer:**
`h = h + Mixer(RMSNorm(h)); h = h + MLP(RMSNorm(h))`.
- `RMSNorm(x) = x * rsqrt(mean(x²) + eps) * (1 + w)`, with `w` initialised to 0.
- Every layer has a SwiGLU MLP: `down(silu(gate(x)) * up(x))`, all without bias.
- Mixer = `Qwen3_5GatedDeltaNet` when `layer_types[i] == "linear_attention"`.

**Gated DeltaNet forward, per token:**
1. `qkv = in_proj_qkv(x)`. This goes through a depthwise causal `conv1d` (kernel 4, **no bias**), then
   **SiLU**, then is split into q, k, v.
2. `z = in_proj_z(x)`, `b = in_proj_b(x)`, `a = in_proj_a(x)`. These are linear and **have no bias**.
3. `beta = sigmoid(b)`; `g = -exp(A_log) * softplus(a + dt_bias)`; the decay is `exp(g)`.
4. q and k are **L2-normalised** (eps 1e-6). Then `q /= sqrt(head_k_dim)`.
5. State update (per head, `S` of shape `[k_dim, v_dim]`):
   `S = exp(g)·S;  S += k ⊗ beta·(v − Sᵀk)` (the delta rule). Output `o = Sᵀq` (read-after-write).
6. `o = RMSNormGated(o) = rmsnorm_per_head(o) * weight * silu(z)`, then `out_proj`.
7. q and k heads are `repeat_interleave`d to the v heads when `num_v_heads > num_k_heads`. **We need
   one independent query per head, so set `num_k_heads == num_v_heads`.**

**Consequences the compiler must handle:**
| Fact | Design rule |
|---|---|
| None of b, a or the conv has a bias | Keep a constant **ONE** dimension in the residual. Every "bias" is a weight on ONE. |
| Exact overwrite needs `beta = 1` | `b = +30` on write and `−30` on no-write. In fp32, `sigmoid(30)` rounds to 1.0. |
| Exact memory needs decay = 1 | `A_log = −40` (exp ≈ 4e-18), so `exp(g)` = 1.0 in fp32. |
| SiLU after the conv | Use conv weight `[0,0,0,1]` (current token only). Drive k, q, v as {0, +L} with L=30. `silu(30) ≈ 30` and `silu(0) = 0`. Never drive them negative. |
| L2 norm on q and k | A one-hot `L·e_i` normalises to exactly `e_i`. |
| Per-token RMSNorm on the residual | **Make the residual's L2 norm identical on every token** (§3.2). Then every RMSNorm is a known constant scale and the circuit is exact. |
| RMSNormGated on reads | Every stored value must have **constant norm** (dual-rail or fixed-count one-hot). Never read a slot that hasn't been written (that returns 0); the boot sequence writes every slot. |
| Chunked prefill uses triangular solves (floating-point error) | Run **token by token with a cache** (the recurrent path). Prefill only BOS. |

`transformers` has pure-PyTorch fallbacks for the conv and the delta rule, so **this runs on CPU or
Apple MPS in fp32. No GPU and no training are needed.** Pin the exact `transformers` version in
`pyproject.toml`; the layer code is the spec.

---

## 2. Game rules (reference engine, `deltris/ref/tetris.py`)

- Board: 10 columns × 20 rows. Row 0 is the top. No hidden rows in v1.
- Pieces: I, O, T, S, Z, J, L. Rotation states are 4 fixed bitmaps per piece inside a 4×4 box (use the
  SRS shapes, but **no kicks**). The piece position is `(x, y)` = the top-left of the 4×4 box, with
  `x ∈ [−2, 9]` and `y ∈ [−2, 19]`. A position is legal iff every filled cell is inside the board
  and on an empty board cell.
- Tick: the host supplies one action ∈ {LEFT, RIGHT, ROT_CW, NOOP}. Apply the action if it's legal (if not,
  ignore it), then apply gravity: `y+1` if legal, otherwise **lock**.
- Lock: OR the piece cells into the board. Full rows are removed and the rows above fall. The line
  count += the number of rows removed.
- Spawn: the host supplies the next piece type. It spawns at `x = 3`, `y = 0` (O at `x = 4`), rotation 0. If
  that position is illegal → **game over**.
- The piece sequence comes from the host (seeded 7-bag in tests). The model contains no RNG.

The reference engine is the oracle. It must be small, obviously correct, and have 100% branch coverage.

---

## 3. The token protocol ("trace ISA")

### 3.1 Memory map (one designated DeltaNet layer, `L_MEM` = layer 0)

The board heads and register heads below live in `L_MEM`. Every write is specified **entirely by the
current token**. The compiled circuit only decides which token comes next.

| Head group | Heads | Slot (key) | Value |
|---|---|---|---|
| BOARD | 20 replicas (B0…B19) | slots 0–19 = rows | row as **dual-rail** 10 cells → 20 dims |
| PIECE register | 1 | slot 0 | one-hot type(7) + rot(4) + x(12) + y(22) |
| PHASE register | 1 | slot 0 | one-hot {BOOT, PLAY, LOCK, CLEAR, SPAWN, OVER} |
| T register | 1 | slot 0 | one-hot target row (−1…19) |
| S register | 1 | slot 0 | one-hot source row (−1…19) |

- A board write goes to **all 20 BOARD replicas** with the same (k, v). Every token, replica *Bi* is
  queried at slot *i* (with writes disabled for that read if the token isn't a board write). So **the whole
  board is read every token**: row *i* arrives from head *Bi*.
- Each register head always has its query on slot 0. Its write is enabled only by its own token type.
- Suggested dimensions: `head_k_dim = 32`, `head_v_dim = 64`, `num_heads = 24`.

### 3.2 Constant-norm rule (the single most important invariant)
- Every **embedding row** has the same L2 norm. Pad with one FILL dimension set to
  `sqrt(C − ‖fields‖²)`.
- Every value a layer writes into the residual goes into **fresh dims** that are **dual-rail or one-hot
  with a fixed active count**. Every layer writes on every token, whatever the token type.
- Result: ‖h‖ is identical on every token at every layer, so each RMSNorm is a known constant.
- Test: assert that ‖h‖ is constant to 1e-5 relative at every layer, over 10k random traces.

### 3.3 Vocabulary
**Host → model:** `BOS`, `ACT_{LEFT,RIGHT,ROT_CW,NOOP}`, `PIECE_{I,O,T,S,Z,J,L}`.

**Model-emitted (each one is one memory write):**
| Token | Count | Writes |
|---|---|---|
| `WROW(r, bits)` | 20 × 1024 | BOARD slot r ← bits; T ← r |
| `SETP(type, rot, x, y)` | 7×4×12×22 | PIECE register |
| `SRC(s)` | 21 | S register |
| `PHASE(p)` | 6 | PHASE register |
| `TICK_END`, `NEED_PIECE`, `GAME_OVER`, `LINES(k)` k∈1..4 | 7 | nothing (control) |

Total ≈ 28.0k tokens. The LM head output is one-hot (logits ±30), and decoding is greedy argmax.

### 3.4 Microprogram (implement it in the reference **ISA interpreter** first)
- **Boot:** after BOS, the model emits `WROW(r, 0)` for r = 19…0, then the register inits, then `NEED_PIECE`.
- **SPAWN:** on `PIECE_p` → `SETP(spawn)` + `PHASE(PLAY)` + `TICK_END`, or `PHASE(OVER)` + `GAME_OVER`.
- **PLAY:** on `ACT_a` → compute the post-action, post-gravity piece. If the piece moved → `SETP(new)` +
  `TICK_END`. If it locks → `PHASE(LOCK)`.
- **LOCK:** emit `WROW(r, board[r] | piece_row(r))` for each row the piece occupies, top to bottom.
  Then, if any row is full: `PHASE(CLEAR)`, `SRC(first non-full row ≤ 19)`, and the compaction loop. Otherwise
  `PHASE(SPAWN)` + `NEED_PIECE`.
- **CLEAR (compaction, bottom-up, safe because sources lie above targets):**
  the next target is `t = T − 1` (start at 19), with the source taken from S. Emit
  `WROW(t, board[S] or 0 if S = −1)`, then `SRC(next non-full row above S, or −1)`. When `t = 0`
  is done → `LINES(k)` + `PHASE(SPAWN)` + `NEED_PIECE`.
- The host loop is: append a host token, then let the model generate until it emits
  `TICK_END | NEED_PIECE | GAME_OVER`.

This microprogram is a **baseline design**. The token count per line clear is a research variable (§7.4).
Any change must first be made in the ISA interpreter and pass the P0 tests.

---

## 4. Phases and acceptance tests

### P0 – Oracles (pure Python, no NN)
- `ref/tetris.py`: the game rules from §2.
- `ref/isa.py`: an interpreter that **executes the token protocol**. It holds the memory map as plain arrays
  and, given the trace so far, deterministically produces the next token.
- **Accept:** on 10 000 seeded random games (random actions, 7-bag), the board after every tick of
  `isa.py` equals `tetris.py`. Include forced cases: an I-piece Tetris (4-line clear), non-contiguous
  clears (rows 17 and 19), top-out, and rotation rejected at a wall.

### P1 – Compiler primitives (`deltris/compiler/`)
Write a small graph DSL that compiles to **real `Qwen3_5ForCausalLM` weights**. Primitives:
- `onehot_field`, `dualrail_field` (residual layout allocator, with a ONE and FILL dim)
- `mem_write(group, slot_from_token, value_from_token, enable_from_token)` and
  `mem_read(group, slot)` → `L_MEM` projections
- `linear_map` (one-hot permutations, e.g. x±1, rot+1 mod 4; these are free and linear)
- `and_k`, `or_k`, `not` via MLP threshold units with margins
- `mux(select_onehot, inputs)` via SwiGLU: `silu(±30 selector) * value`
- `lut(onehot_in → bits_out)` (piece-shape tables)
- `emit(token_fields)` → LM head
- `noop_mixer` for DeltaNet layers that aren't used (out_proj = 0)

**Accept:** each primitive is compiled in isolation into a 2–3 layer model and run through the stock HF
forward with a cache. Its outputs match a NumPy reference **after binarisation**, with every margin ≥ 0.4.
Tests for every row of the §1 table (beta = 1.0 exactly, decay = 1.0 exactly, exact overwrite after 10⁶
writes to the same slot).

### P2 – Full compile
- `compile_deltris() → (config, state_dict)`. Save it with `save_pretrained`.
  `AutoModelForCausalLM.from_pretrained` must load it with **no custom code**.
- Use all-`linear_attention` layer types. Pick the hidden size and layer count yourself. Expect roughly
  1–3k dims and 10–20 layers, and write the final numbers down in `DECISIONS.md`.
- **Accept (differential test):** on 10 000 seeded games, the model's generated trace equals
  `isa.py`'s trace **token for token**. Also, after every token, the board decoded from the real
  recurrent state (`cache.get_recurrent_state(L_MEM)`) equals the oracle. Run it in fp32.

### P3 – Visuals (the eye-catching part)
- `play.py`: terminal (curses) play with arrow keys. The board is drawn **only from the decoded DeltaNet
  state**, never from Python game state.
- `record.py` → MP4 or GIF with three panels: (1) the board decoded from the state, (2) a raw heatmap
  of the BOARD head's `[k_dim × v_dim]` state matrix (the board should be visibly *in* it), (3) the scrolling
  token trace with the memory writes highlighted.
- **Accept:** a 60-second recording that includes at least one multi-line clear.

### P4 – Robustness
- Long-run test: a 1M-token game with no divergence from the oracle.
- bf16 run: report the first divergence, if there is one. Don't try to fix it in v1; just write it down.
- Rotation-invariance check (needed for §7): apply a random orthogonal rotation to the residual stream
  (`E ← E·R`, every input projection `W ← Rᵀ·W`, every output projection `W ← W·R`; the RMSNorm
  weights are all 1, so this is exact). Permute the MLP hidden units randomly. **Accept:** the
  differential test still passes 100%. Ship both `deltris-clean` and `deltris-rotated`.

---

## 5. Repo layout
```
deltris/
  ref/        tetris.py, isa.py
  compiler/   layout.py, primitives.py, graph.py, emit_hf.py
  build.py    -> checkpoints/deltris-clean, checkpoints/deltris-rotated
  play.py  record.py  difftest.py
tests/        test_ref.py, test_isa.py, test_primitives.py, test_arch_facts.py, test_diff.py
DECISIONS.md  PROGRESS.md  README.md  pyproject.toml   (torch, transformers==<pinned>, numpy, pytest)
```

## 6. Rules for the implementing agent
- Phases are strictly ordered: P0 → P4. Paste each phase's acceptance output into `PROGRESS.md`.
- Correctness is defined by the oracles. Never edit a test so that it passes.
- When something is ambiguous, pick the simplest option that keeps §3.2 true, write it down in `DECISIONS.md`, and keep going.
- **Stop and ask the human** if a §1 architecture fact turns out to be false in the pinned
  `transformers` version. The whole design depends on those facts.

---

## 7. Research protocol: weight-space programming with ground truth

This is the reason for building Deltris. The Sparks post grades by behaviour. We can grade by
**construction**, because we know every circuit.

### 7.1 Isolation
- The subject agent runs in a separate sandbox **with no access to this spec, the compiler source,
  the repo, or the internet**. It gets only what the task lists.
- Every task is run at forward-pass budgets **{0, 10, 50}**, which mirrors the post. A "forward pass" is
  one call to a provided `run(tokens) → logits, states` tool, and each call is counted and logged.
- Every task is run on **both** `deltris-clean` and `deltris-rotated` (§P4).

### 7.2 READ tasks (given: safetensors + config.json, nothing else)
Graded by rubric against ground truth:
1. Is this a trained model or a constructed program? Why?
2. What does it compute? (any game → Tetris → board size and orientation)
3. Which layer and heads are memory, and what is the slot ↔ row mapping?
4. What do the token families mean? (WROW, SETP, …)
5. How does line clearing work?

### 7.3 FIX tasks (given: weights + the §2 rules + one failing trace)
Ground-truth corruptions, each with a known minimal fix:
- (a) one wrong cell in the T-piece rotation-2 LUT
- (b) compaction skips at most 1 row (multi-line clears break)
- (c) right-wall collision off by one
Grade: differential-test pass rate after the edit, number of weights changed versus the minimal ground-truth
diff, and whether the edit is in the right component.

### 7.4 WRITE tasks (given: the §2 rules + §3 protocol + the §1 architecture facts + an empty config)
- W1: the full protocol as specified. Grade: differential-test pass rate.
- W2: the rules only, with the protocol left open. The subject designs its own ISA. Grade by an oracle that checks
  **game behaviour** (it decodes the board through the subject's declared memory map). Also record
  tokens per line clear, layer count, and nonzero weights, and compare them with our baseline.

### 7.5 The headline comparison
| | clean | rotated |
|---|---|---|
| READ score | ? | ? |
| FIX score | ? | ? |

**Hypothesis:** an LLM reads compiled weights mainly through **basis-aligned sparsity**, so READ
and FIX should collapse on `deltris-rotated` even though the function is identical. If they don't
collapse, the subject is reading the actual structure, which is a much stronger claim than anything
in the Sparks post. Trained networks are never basis-aligned, so the rotated variant is the
honest proxy for the case that matters.

---

## 8. Stretch goals (only after P4)
- SRS wall kicks, hold, hard drop (a hard drop is a loop of gravity tokens, which is a nice extra microprogram).
- In-model 7-bag RNG (a xorshift in a register head).
- Graft Deltris into Qwen3.5-4B as extra layers and dims, the way Qwenputer does, so that the language model can
  call it with "start Tetris".
- Fewer tokens per line clear: write several rows per token using several BOARD groups.
