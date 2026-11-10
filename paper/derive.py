"""Derive every number and figure in the paper from result files.

    uv run python paper/derive.py

Inputs: results/p2-games.jsonl (P2 differential test, one row per game),
results/deltris-p3.pkl (P3 rollout), models/deltris (the compiled checkpoint),
deltris/ref/isa.py (vocabulary). Outputs: paper/numbers.tex (one macro per
number), paper/fig-state.tex and paper/fig-gaps.tex (TikZ).

Values that exist only in a job log are copied in LOG below with their source.
"""

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper"

# observation: copied from the job logs quoted in PROGRESS.md (P0, P1, P2, P3).
LOG = {
    "PtwoSeconds": ("1518", "PROGRESS.md P2: RESULT games=10500 failing=0 seconds=1518"),
    "PtwoResidualErr": ("3.22e-06", "PROGRESS.md P2: compiled: 186 tokens, max residual error 3.22e-06"),
    "PzeroGreedyMulti": ("1{,}348", "PROGRESS.md P0: 18348 lines, 1348 multi-line clears"),
    "PoneTests": ("39", "PROGRESS.md P1: 39 passed"),
    "PoneOneGate": ("1.2e-7", "PROGRESS.md P1: a one-gate probe measured 1.2e-7"),
    "PthreeRolloutSeconds": ("18", "PROGRESS.md P3: rollout 1873 tokens in 18s"),
    "PthreeLaptopMs": ("135", "PROGRESS.md P3: about 135 ms per token"),
}


def fmt(n):
    return f"{n:,}".replace(",", "{,}")


def main():
    m = {}

    # ---- P2 differential test -------------------------------------------
    rows = [json.loads(l) for l in open(ROOT / "results/p2-games.jsonl")]
    m["PtwoGames"] = fmt(len(rows))
    sets = {}
    for r in rows:
        sets.setdefault(r["game"].split("-")[0], []).append(r)
    for name, key in (("random", "Rand"), ("greedy", "Greedy")):
        g = sets[name]
        m[f"Ptwo{key}Games"] = fmt(len(g))
        m[f"Ptwo{key}Tokens"] = fmt(sum(r["tokens"] for r in g))
        m[f"Ptwo{key}Checked"] = fmt(sum(r["checked"] for r in g))
        m[f"Ptwo{key}Lines"] = fmt(sum(r["lines"] for r in g))
        m[f"Ptwo{key}Gap"] = f"{min(r['min_logit_gap'] for r in g):.4f}"
        m[f"Ptwo{key}TokFail"] = fmt(sum(r["first_token_mismatch"] >= 0 for r in g))
        m[f"Ptwo{key}BoardFail"] = fmt(sum(r["first_board_mismatch"] >= 0 for r in g))
    m["PtwoTokens"] = fmt(sum(r["tokens"] for r in rows))
    m["PtwoChecked"] = fmt(sum(r["checked"] for r in rows))
    m["PtwoTokFail"] = fmt(sum(r["first_token_mismatch"] >= 0 for r in rows))
    m["PtwoBoardFail"] = fmt(sum(r["first_board_mismatch"] >= 0 for r in rows))
    gaps = np.array([r["min_logit_gap"] for r in rows])
    m["PtwoGap"] = f"{gaps.min():.3f}"
    m["PtwoGapMedian"] = f"{np.median(gaps):.4f}"
    m["PtwoMaxGameTokens"] = fmt(max(r["tokens"] for r in rows))

    # ---- checkpoint ------------------------------------------------------
    cfg = json.load(open(ROOT / "models/deltris/config.json"))
    sd = load_file(ROOT / "models/deltris/model.safetensors")
    nparam = sum(v.size for v in sd.values())
    nnz = sum(int(np.count_nonzero(v)) for v in sd.values())
    m["Params"] = fmt(nparam)
    m["ParamsM"] = f"{nparam / 1e6:.1f}"
    m["Nonzero"] = fmt(nnz)
    vocab_params = sd["model.embed_tokens.weight"].size + sd["lm_head.weight"].size
    m["VocabParamsM"] = f"{vocab_params / 1e6:.1f}"
    m["VocabParamsPct"] = f"{100 * vocab_params / nparam:.0f}"
    m["NonzeroPct"] = f"{100 * nnz / nparam:.2f}"
    m["Hidden"] = fmt(cfg["hidden_size"])
    m["Layers"] = str(cfg["num_hidden_layers"])
    m["Inter"] = fmt(cfg["intermediate_size"])
    m["Heads"] = str(cfg["linear_num_value_heads"])
    m["Kdim"] = str(cfg["linear_key_head_dim"])
    m["Vdim"] = str(cfg["linear_value_head_dim"])
    m["Vocab"] = fmt(cfg["vocab_size"])
    m["TfVersion"] = cfg["transformers_version"]
    units = []
    for i in range(cfg["num_hidden_layers"]):
        g = sd[f"model.layers.{i}.mlp.gate_proj.weight"]
        units.append(int(np.count_nonzero(np.abs(g).sum(1))))
    for i, u in enumerate(units):
        m[f"Units{'ABCDEF'[i]}"] = fmt(u)
    m["UnitsTotal"] = fmt(sum(units))
    emb = sd["model.embed_tokens.weight"].astype(np.float64)
    norms = np.linalg.norm(emb, axis=1)
    m["EmbNormSpread"] = f"{norms.max() - norms.min():.1e}"

    # ---- vocabulary ------------------------------------------------------
    import sys
    sys.path.insert(0, str(ROOT))
    from deltris.ref import isa
    fam = Counter(t[0] for t in isa.vocabulary())
    for k, name in (("WROW", "Wrow"), ("SETP", "Setp"), ("SRC", "Src"), ("PHASE", "Phase"),
                    ("ACT", "Act"), ("PIECE", "Piece"), ("LINES", "Lines")):
        m[f"Voc{name}"] = fmt(fam[k])
    m["VocOther"] = fmt(sum(v for k, v in fam.items()
                            if k not in ("WROW", "SETP", "SRC", "PHASE", "ACT", "PIECE", "LINES")))

    # ---- P3 rollout ------------------------------------------------------
    d = pickle.load(open(ROOT / "results/deltris-p3.pkl", "rb"))
    rec = d["rec"]
    m["PthreeSeed"] = str(d["seed"])
    m["PthreeTokens"] = fmt(len(rec))
    m["PthreeGap"] = f"{min(r['gap'] for r in rec if r['gap'] is not None):.3f}"
    lines_at = [i for i, r in enumerate(rec) if r["tok"][0] == "LINES"]
    multi = [i for i in lines_at if rec[i]["tok"][1] >= 2]
    m["PthreeMultiTok"] = str(multi[0])
    m["PthreeMultiK"] = str(rec[multi[0]]["tok"][1])
    clear = multi[0] - 1
    assert rec[clear]["tok"] == ("PHASE", "CLEAR")
    m["PthreeClearTok"] = str(clear)
    m["PthreeLinesTotal"] = str(sum(rec[i]["tok"][1] for i in lines_at))

    for k, (v, _) in LOG.items():
        m[k] = v

    with open(OUT / "numbers.tex", "w") as f:
        f.write("% generated by paper/derive.py; do not edit\n")
        for k in sorted(m):
            src = f"  % {LOG[k][1]}" if k in LOG else ""
            f.write(f"\\newcommand{{\\n{k}}}{{{m[k]}}}{src}\n")

    write_state_figure(rec[clear])
    write_gap_figure(gaps)
    print(f"{len(m)} numbers; units per layer {units}")


def write_state_figure(r):
    """DISPLAY state matrix next to the decoded board, at the PHASE(CLEAR) token."""
    disp = np.asarray(r["display"], dtype=np.float64)
    board = r["board"]
    c = 0.16  # cm per cell (cosmetic)
    out = ["% generated by paper/derive.py from results/deltris-p3.pkl; do not edit",
           "\\begin{tikzpicture}"]
    vmax = np.abs(disp).max()
    k_dim, v_dim = disp.shape
    for i in range(k_dim):
        for j in range(v_dim):
            v = disp[i, j] / vmax
            shade = int(round(100 * max(v, 0)))
            out.append(f"\\fill[black!{shade}!white] ({j * c:.3f},{-i * c:.3f}) rectangle +({c:.3f},{-c:.3f});")
    out.append(f"\\draw[gray] (0,0) rectangle ({v_dim * c:.3f},{-k_dim * c:.3f});")
    out.append(f"\\node[below] at ({v_dim * c / 2:.3f},{-k_dim * c:.3f}) {{\\small (a) DISPLAY state, {k_dim}$\\times${v_dim}}};")
    x0 = v_dim * c + 1.0
    for rr, bits in enumerate(board):
        for col in range(10):
            on = bits >> col & 1
            fill = "black!70" if on else "black!6"
            out.append(f"\\fill[{fill}] ({x0 + col * c:.3f},{-rr * c:.3f}) rectangle +({c * 0.9:.3f},{-c * 0.9:.3f});")
    for rr in (17, 18):
        out.append(f"\\draw[thick] ({x0 - 0.03:.3f},{-rr * c + 0.02:.3f}) rectangle +({10 * c + 0.03:.3f},{-c:.3f});")
    out.append(f"\\node[below] at ({x0 + 5 * c:.3f},{-k_dim * c:.3f}) {{\\small (b) decoded board}};")
    out.append("\\end{tikzpicture}")
    (OUT / "fig-state.tex").write_text("\n".join(out) + "\n")


def write_gap_figure(gaps):
    """Histogram of the per-game smallest logit gap."""
    lo, hi, nb = 29.975, 30.0, 25  # cosmetic: bin range covers every observed value
    assert gaps.min() >= lo and gaps.max() <= hi
    counts, edges = np.histogram(gaps, bins=nb, range=(lo, hi))
    w, h = 10.0, 4.0  # cm (cosmetic)
    top = counts.max()
    out = ["% generated by paper/derive.py from results/p2-games.jsonl; do not edit",
           "\\begin{tikzpicture}"]
    for i, n in enumerate(counts):
        x = w * i / nb
        out.append(f"\\fill[black!55] ({x:.3f},0) rectangle +({w / nb * 0.85:.3f},{h * n / top:.3f});")
    out.append(f"\\draw (0,0) -- ({w},0);")
    for v in (29.975, 29.98, 29.985, 29.99, 29.995, 30.0):
        x = w * (v - lo) / (hi - lo)
        out.append(f"\\draw ({x:.3f},0) -- +(0,-0.1) node[below] {{\\scriptsize {v:g}}};")
    out.append(f"\\node at ({w / 2},-0.85) {{\\small smallest logit gap in the game}};")
    out.append(f"\\draw (0,0) -- (0,{h});")
    out.append(f"\\node[left] at (0,{h}) {{\\scriptsize {top:,}}};")
    out.append("\\node[left] at (0,0) {\\scriptsize 0};")
    out.append(f"\\node[rotate=90] at (-0.9,{h / 2}) {{\\small games}};")
    out.append("\\end{tikzpicture}")
    (OUT / "fig-gaps.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
