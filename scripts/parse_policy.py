"""Deterministic chunker: Export Control Rules 2021 PDF -> one markdown file per section.
Split on the body heading pattern '<section-id>  <Title>\n' (two-space separator distinguishes
body headings from table-of-contents entries, which use a single space + newline + dot leaders)."""
import pymupdf, re, os, json

ROOT = "/Users/siddhantparashar/projects/cyberai_task2"
PDF = f"{ROOT}/Task2/1-Export Control (Plants and Plant Products)Rules 2021.pdf"
OUT = f"{ROOT}/policy"

doc = pymupdf.open(PDF)
full = "\n".join(p.get_text() for p in doc)

HEAD = re.compile(r'\n(\d+-\d+[A-Z]?)  ([A-Z][^\n]{2,90})\n')
matches = list(HEAD.finditer(full))
assert len(matches) > 150, f"chunker regression: only found {len(matches)} section headings"

sections = []
for i, m in enumerate(matches):
    sec_id = m.group(1)
    start = m.start() + 1  # skip leading \n
    end = matches[i+1].start()+1 if i+1 < len(matches) else len(full)
    body = full[start:end].strip()
    # first line (after the id) is the section title, possibly wrapped across lines
    title_line = body.split("\n", 1)[0]
    title = re.sub(r'^' + re.escape(sec_id) + r'\s+', '', title_line).strip()
    sections.append({"id": sec_id, "title": title, "text": body})

os.makedirs(OUT, exist_ok=True)
index = []
for s in sections:
    fname = f"section_{s['id']}.md"
    path = os.path.join(OUT, fname)
    with open(path, "w") as fh:
        fh.write(f"# Section {s['id']} — {s['title']}\n\n{s['text']}\n")
    index.append({"id": s["id"], "title": s["title"], "file": f"policy/{fname}", "chars": len(s["text"])})

json.dump(index, open(f"{OUT}/_index.json", "w"), indent=1)
with open(f"{OUT}/_index.md", "w") as fh:
    fh.write("# Policy section index — Export Control (Plants and Plant Products) Rules 2021\n\n")
    fh.write("Deterministically chunked from the PDF, one file per section. Source of truth for OKF citations.\n\n")
    for s in index:
        fh.write(f"- **{s['id']}** — {s['title']}  (`{s['file']}`, {s['chars']} chars)\n")

print(f"wrote {len(sections)} section files to {OUT}/")
print("sample ids:", [s["id"] for s in sections[:5]], "...", [s["id"] for s in sections[-5:]])
