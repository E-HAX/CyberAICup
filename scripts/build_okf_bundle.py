"""Deterministically render the OKF bundle: one cps/cp_XX.md file per checking point,
with YAML frontmatter citing the exact policy/section_*.md files that govern it
(as determined by the grounded-reading mapping agents, not hard-coded here)."""
import json, glob, os, re

ROOT = "/Users/siddhantparashar/projects/cyberai_task2"
CPS = json.load(open(f"{ROOT}/dev/cp_reference.json"))  # official text, element, subelement, in order
CP_BY_CODE = {c["cp"]: c for c in CPS}

mapping = {}
for f in sorted(glob.glob(f"{ROOT}/cps_mapping/element*.json")):
    for entry in json.load(open(f)):
        mapping[entry["cp"]] = entry

os.makedirs(f"{ROOT}/cps", exist_ok=True)

def yaml_list(items):
    return "\n".join(f"  - {it}" for it in items)

index_lines = ["# OKF bundle index — 41 checking points mapped to governing policy sections\n",
               "Each cp_XX.md pairs the official checking-point text with the exact policy/section_*.md",
               "file(s) that were read and confirmed to govern it. Generated from cps_mapping/element*.json",
               "(produced by grounded-reading mapping agents), not hand-authored.\n"]

for cp in CPS:
    code = cp["cp"]
    num = int(re.match(r'CP(\d+)', code).group(1))
    m = mapping.get(code)
    if not m:
        raise SystemExit(f"missing mapping for {code}")
    citations = m["citations"]
    rationale = m["rationale"]

    fname = f"cp_{num:02d}.md"
    fm = (
        "---\n"
        f"cp: {code}\n"
        f"element: \"{cp['element']}\"\n"
        f"subelement: \"{cp['subelement']}\"\n"
        f"citations:\n{yaml_list(citations)}\n"
        "---\n\n"
    )
    body = (
        f"# {code} — {cp['subelement']}\n\n"
        f"**Official checking-point text:**\n{cp['text']}\n\n"
        f"**Why these sections govern this checking point:**\n{rationale}\n\n"
        f"**Governing policy text** (concatenated from the cited sections above):\n\n"
    )
    for c in citations:
        sec_text = open(f"{ROOT}/{c}").read()
        body += f"---\n\n{sec_text}\n"

    open(f"{ROOT}/cps/{fname}", "w").write(fm + body)
    index_lines.append(f"- **{code}** ({cp['element']} — {cp['subelement']}) → `{fname}` — cites {', '.join(citations)}")

open(f"{ROOT}/cps/_index.md", "w").write("\n".join(index_lines) + "\n")
print(f"wrote 41 cp_XX.md files + cps/_index.md")
