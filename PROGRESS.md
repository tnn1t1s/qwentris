# Progress

## P0 – Oracles: passed (2026-11-03)

`uv run pytest -q -s tests/ --cov=deltris.ref --cov-branch`

```
10000 random games to top-out, 61 lines
500 greedy games x 100 pieces, 18348 lines, 1348 multi-line clears
vocabulary size 35309

Name                      Stmts   Miss Branch BrPart  Cover   Missing
deltris/ref/isa.py          125      1     64      1    99%   165
deltris/ref/players.py       39      1     18      1    96%   51
deltris/ref/tetris.py        83      0     28      0   100%
30 passed in 35.69s
```

`tetris.py` has 100% branch coverage. `isa.py` line 165 is the unreachable
`raise` for an undefined transition. Forced cases: four-line clear with residue
above the stack, non-contiguous clear (rows 17 and 19), top-out, rotation
rejected at the right wall, and lock after a legal slide (D3).

The protocol differs from SPEC §3.4 in D2–D5 (DECISIONS.md).

## P1 – Compiler primitives: passed (2026-11-03)

`uv run pytest -q -s tests/` (39 tests, 81.5 s on the laptop, one core)

```
tests/test_ref.py          10 passed
tests/test_isa.py           9 passed
tests/test_arch_facts.py   11 passed
tests/test_primitives.py    9 passed
memory: 421 tokens, margin 0.500000, 10-bit rows
39 passed in 81.50s
```

Each primitive is compiled alone into a stock `Qwen3_5ForCausalLM` and run token
by token with a `DynamicCache`. The checks:

- `threshold` / `and_` / `or_` / `not_`: all 64 inputs of a 6-bit field, two
  levels deep (the second level reads first-level outputs, including a NOT view).
- `lut` / `linear_map` / `relabel`: every (type, rot, x): the piece-shape table,
  shape after rot+1, and x+1 with an out-of-range label.
- `mux`: 300 random selections over five 10-bit fields and one constant.
- `memory`: 20 single-slot row heads, a 20-slot display head and a onehot
  register over 421 random writes and reads. Every read equals the last write,
  and the board decoded from the display head's recurrent state equals the
  reference board.
- `emit`: 560 inputs over a 20 480-token WROW-style vocabulary; argmax equals
  the reference token every time.
- beta is exactly 1.0 on writes and < 1e-12 otherwise; decay is exactly 1.0.
- 10⁶ alternating writes to one slot, fed from the compiled layer-0 projections straight into the HF recurrent kernel (`torch_recurrent_gated_delta_rule`; a full forward per token is estimated at about 15 min): the slot
  holds the last value, an untouched slot holds its boot value.
- `save_pretrained` then `AutoModelForCausalLM.from_pretrained` returns
  `Qwen3_5ForCausalLM` with every tensor bit-identical (no re-initialisation of
  `A_log` or `dt_bias`).

Every test asserts margin ≥ 0.4 and residual norm constant to 1e-5 relative
at every layer. Observed: the memory test printed margin 0.500000 (residual error under 5e-7); a one-gate probe measured 1.2e-7.

## P2 – Compiled Tetris: passed (2026-11-03)

Run on one NVIDIA GB10 GPU (fp32, TF32 off), container image
`nvidia/cuda:13.0.2-base-ubuntu24.04`, torch 2.15.0+cu132, transformers 5.22.1.
Command: `python -m deltris.difftest --random 10000 --greedy 500`; per-game results: `results/p2-games.jsonl`
(sha256 `bcc0f7d5…70e6d`).

```
tests/test_deltris.py       4 passed in 87.28s   (same job, CPU)
  reference: 50 random + 5 greedy games, 184 lines
  compiled: 186 tokens, max residual error 3.22e-06
wrote model: hidden 1681, layers 6, intermediate 1860, heads 25, vocab 35309
10500/10500 games, 0 failing, 1518s
RESULT games=10500 failing=0 seconds=1518

random 10000 games 6502442 tokens 4607950 checked    61 lines  token-fail 0  board-fail 0  min gap 29.9812
greedy   500 games 3629930 tokens 2747880 checked 18348 lines  token-fail 0  board-fail 0  min gap 29.9792
```

The model is a stock `Qwen3_5ForCausalLM` saved with `save_pretrained` and
loaded with `AutoModelForCausalLM`, with no custom code. Each game runs
token by token through the cached recurrent path, with the isa trace fed in
(D14). Over 10 132 372 tokens:

- every one of the 7 355 830 model-emitted tokens is the argmax and equals the
  isa token;
- after every token, the board decoded from the DISPLAY head's recurrent state
  equals the oracle board;
- the smallest logit gap between the correct token and the next is 29.979
  (design 30).

The line totals equal P0's (61 random, 18 348 greedy), as they must, since the
same seeds are used.

## P3 – Play and record: passed (2026-11-03)

Run on one NVIDIA GB10 GPU (fp32, TF32 off) with
`python -m deltris.record --model $MODEL --device cuda --out $OUT/deltris-p3.mp4`.
The video (1280×720, 15 fps) is not in the repository; `python -m deltris.record`
regenerates it from the same seed. The rollout it draws from is `results/deltris-p3.pkl`.

```
seed 10: first 2-line clear done at 26.1s of video
rollout 1873 tokens in 18s, min gap 29.983
rollout equals the isa trace and oracle board
RESULT frames=900 seconds=60.0 multi_line_clears=1 render=11s
```

- The model plays the game on its own. The host sends PIECE and ACT tokens
  from a greedy planner that reads the board and the falling piece decoded from
  state. The model emits every other token.
- The 1873-token trace is a valid isa trace. After every token, the board
  decoded from the DISPLAY head equals the board rebuilt from the WROW tokens.
- Each frame has four panels: the board from the DISPLAY state with the
  falling piece taken from the PIECE register; the raw DISPLAY state matrix
  [20 keys × 45 values]; the registers; and the token trace, with memory
  writes colored and the written row outlined.
- Frames fall at tick ends and on every token from a lock to the next
  NEED_PIECE, so each line clear plays out row by row. The 4-row lock at token
  831 completes rows 17 and 18. A 2-line clear follows (LINES 2 at token 832),
  and the stack shifts down two rows by SRC/WROW.
- `python -m deltris.play --model models/deltris --device cpu` runs the same
  host loop in curses with arrow keys, drawn only from decoded state. A
  headless run of the loop with scripted moves on the laptop CPU measured
  about 135 ms per token, roughly 0.4 s per tick.
