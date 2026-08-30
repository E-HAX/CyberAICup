import json, re, os
ROOT="/Users/siddhantparashar/projects/cyberai_task2"
template = open(f"{ROOT}/dev/dashboard/template.html").read()
data_raw = open(f"{ROOT}/dev/dashboard/data.json").read()
# safe to embed inside a <script type="application/json"> block: only need to guard against "</script"
safe = data_raw.replace("</script", "<\\/script")
out = template.replace("__DATA_JSON__", safe)
outpath = f"{ROOT}/dev/dashboard/index.html"
open(outpath, "w").write(out)
print("wrote", outpath, f"({os.path.getsize(outpath)/1024:.0f} KB)")
