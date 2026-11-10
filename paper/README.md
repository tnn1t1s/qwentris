# Paper

`main.tex` is the paper. Every number in it is a macro from `numbers.tex`, and both figures are TikZ files; `derive.py` writes all three from `results/` and `models/deltris`. `claims.md` lists each claim with its type and the artifact behind it.

Build, from the repository root:

```sh
uv run python paper/derive.py
cd paper && tectonic main.tex
```

References are built from fetched metadata, not typed. To refetch:

```sh
curl -sL "https://export.arxiv.org/api/query?id_list=2412.06464,2406.06484,2102.11174,2106.06981,2301.05062,2301.13196,2505.09388,2002.05202,1702.03118&max_results=20" -o paper/arxiv.xml
curl -sL https://api.github.com/repos/divagr18/qwenputer -o paper/qwenputer.json
uv run python paper/bib.py
```
