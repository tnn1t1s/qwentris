"""Write paper/references.bib from arXiv API metadata (paper/arxiv.xml), the GitHub API
records for Qwenputer and nesformer, and the citation meta tags of the Sparks blog post.
Fetch: see the curl lines in paper/README.md."""
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
for key, name in (("qwenputer", "Qwenputer"), ("nesformer", "nesformer")):
    g = json.load(open(P / f"{key}.json"))
    out.append(f"@misc{{{key},\n  title={{{{{name}}}: {g['description']}}},\n  author={{{g['owner']['login']}}},\n"
               f"  year={{{g['created_at'][:4]}}},\n  howpublished={{\\url{{{g['html_url']}}}}}\n}}\n")
# citation_* meta tags of the Sparks blog post; fetch command in paper/README.md
meta = re.findall(r'<meta name="citation_(\w+)" content="([^"]*)">', (P / "sparks-meta.html").read_text())
one = lambda k: next(v for kk, v in meta if kk == k)
authors = " and ".join(v for k, v in meta if k == "author")
out.append(f"@misc{{takehi2026sparks,\n  title={{{{{one('title').replace('&amp;', chr(92) + '&')}}}}},\n"
           f"  author={{{authors}}},\n  year={{{one('publication_date')[:4]}}},\n"
           f"  howpublished={{\\url{{{one('fulltext_html_url')}}}}}\n}}\n")
(P / "references.bib").write_text("\n".join(out))
print(len(out), "entries")
