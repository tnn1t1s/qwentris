# Claim ledger (working file, not typeset)

| # | Claim | Type | Artifact | Scope |
|---|---|---|---|---|
| 1 | The checkpoint is a stock `Qwen3_5ForCausalLM` that loads with `AutoModelForCausalLM` and no custom code | observation | PROGRESS.md P1, P2; models/deltris/config.json | transformers 5.22.1 |
| 2 | No weight is trained; every weight is set by `compile_deltris()` | invariant | deltris/build.py, deltris/compiler/ | by construction |
| 3 | Model size: hidden 1681, 6 layers, intermediate 1860, 25 heads, vocab 35,309, 219.7M params, 1.88M nonzero | observation | models/deltris (derive.py) | one compile |
| 4 | beta = 1.0 on writes, decay = 1.0, silu(30)/30 = 1.0, silu(0) = 0 in fp32 | observation | tests/test_primitives.py, tests/test_arch_facts.py (PROGRESS.md P1) | fp32, CPU |
| 5 | Residual norm is identical on every token at every layer | invariant (checked) | graph.py norm2_at; tests assert 1e-5 relative | P1 tests, P2 CPU test |
| 6 | 10,500 games, 10,132,372 tokens, 7,355,830 checked, 0 token and 0 board mismatches | observation | results/p2-games.jsonl | one run, GB10 fp32 TF32 off, seeds fixed, trace fed (D14) |
| 7 | Smallest logit gap 29.979 against design 30 | observation | results/p2-games.jsonl | same run |
| 8 | Line totals equal the reference (61 random, 18,348 greedy) | observation | p2-games.jsonl, PROGRESS.md P0 | same seeds |
| 9 | Free rollout of 1,873 tokens with a 2-line clear at token 832 equals the isa trace | observation | results/deltris-p3.pkl, PROGRESS.md P3 | seed 10, one game, GB10 |
| 10 | The DISPLAY head state matrix is the board | invariant (checked) | D12; P1 memory test; P2 board check | as 6 |
| 11 | Depth 6 is the ACT-path length under ASAP scheduling | observation | DECISIONS.md D13; build.py | this circuit |
| 12 | Free-run equivalence of the fed-trace test | invariant | D14 argument | holds when every check passes |
| 13 | 1M-token run, bf16, rotation invariance | research question | none (P4) | not run |
| 14 | An LLM reads compiled weights through basis-aligned sparsity | hypothesis | none (SPEC §7.5) | not tested |
| 15 | generate() works unmodified | research question | D11 | not tested |
