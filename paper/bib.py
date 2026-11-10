"""Write paper/references.bib from arXiv API metadata (paper/arxiv.xml) plus the GitHub
API record for Qwenputer. Fetch: see the curl lines in paper/README.md."""
import json, re, xml.etree.ElementTree as E
from pathlib import Path
P = Path(__file__).resolve().parent
ns = {"a": "http://www.w3.org/2005/Atom"}
KEYS = {"2412.06464": "yang2024gated", "2406.06484": "yang2024parallelizing", "2102.11174": "schlag2021linear",
        "2106.06981": "weiss2021thinking", "2301.05062": "lindner2023tracr", "2301.13196": "giannou2023looped",
        "2505.09388": "qwen3report", "2002.05202": "shazeer2020glu", "1702.03118": "elfwing2017silu"}
out = []
for e in E.parse(P / "arxiv.xml").getroot().findall("a:entry", ns):
    aid = re.search(r"abs/(\d+\.\d+)", e.find("a:id", ns).text).group(1)
    title = " ".join(e.find("a:title", ns).text.split())
    authors = " and ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns))
    year = e.find("a:published", ns).text[:4]
    out.append(f"@misc{{{KEYS[aid]},\n  title={{{{{title}}}}},\n  author={{{authors}}},\n  year={{{year}}},\n"
               f"  eprint={{{aid}}},\n  archivePrefix={{arXiv}},\n  url={{https://arxiv.org/abs/{aid}}}\n}}\n")
g = json.load(open(P / "qwenputer.json"))
out.append(f"@misc{{qwenputer,\n  title={{{{Qwenputer}}: {g['description']}}},\n  author={{{g['owner']['login']}}},\n"
           f"  year={{{g['created_at'][:4]}}},\n  howpublished={{\\url{{{g['html_url']}}}}}\n}}\n")
(P / "references.bib").write_text("\n".join(out))
print(len(out), "entries")
