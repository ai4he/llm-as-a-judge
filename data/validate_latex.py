#!/usr/bin/env python3
"""Static validator for paper/main.tex (run when no TeX engine is available).
Checks: citations resolve to references.bib; \\includegraphics targets exist;
\\input targets exist; \\ref targets have \\label; \\begin/\\end balance."""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT/"paper"; FIG = ROOT/"figures"
main = (PAPER/"main.tex").read_text()

def strip_comments(s): return re.sub(r"(?<!\\)%.*", "", s)

# gather text from main + all \input'd files (one level + nested wrappers)
def gather(text, seen=None):
    seen = seen or set(); full = text
    for inc in re.findall(r"\\input\{([^}]*)\}", text):
        p = PAPER/(inc if inc.endswith(".tex") else inc+".tex")
        if p.exists() and str(p) not in seen:
            seen.add(str(p)); full += "\n" + gather(p.read_text(), seen)
    return full
ALL = strip_comments(gather(main))
mains = strip_comments(main)
err = []

# 1. citations
bibkeys = set(re.findall(r"@\w+\{([^,]+),", (PAPER/"references.bib").read_text()))
cited = set()
for m in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]*)\}", ALL):
    for k in m.split(","): cited.add(k.strip())
miss_c = sorted(c for c in cited if c and c not in bibkeys)
print(f"[cite] {len(cited)} unique keys cited; bib has {len(bibkeys)}; missing: {len(miss_c)}")
if miss_c: err.append("missing bib keys: "+", ".join(miss_c)); print("   ->", miss_c)

# 2. graphics exist (pdf or png)
imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", ALL)
miss_i = [g for g in imgs if not (FIG/(g+".pdf")).exists() and not (FIG/(g+".png")).exists()
          and not (PAPER/g).exists()]
print(f"[graphics] {len(imgs)} includes; missing: {len(miss_i)}")
if miss_i: err.append("missing figures: "+", ".join(miss_i)); print("   ->", miss_i)

# 3. inputs exist
inps = re.findall(r"\\input\{([^}]*)\}", ALL)
miss_n = [i for i in inps if not (PAPER/(i if i.endswith('.tex') else i+'.tex')).exists()]
print(f"[input] {len(inps)} inputs; missing: {len(miss_n)}")
if miss_n: err.append("missing inputs: "+", ".join(miss_n)); print("   ->", miss_n)

# 4. refs have labels
labels = set(re.findall(r"\\label\{([^}]*)\}", ALL))
refs = set()
for m in re.findall(r"\\(?:ref|autoref|eqref|cref|Cref)\{([^}]*)\}", ALL):
    for k in m.split(","): refs.add(k.strip())
miss_r = sorted(r for r in refs if r not in labels)
print(f"[ref] {len(refs)} refs; {len(labels)} labels; dangling: {len(miss_r)}")
if miss_r: err.append("dangling refs: "+", ".join(miss_r)); print("   ->", miss_r)

# 5. environment balance (main.tex only; ignore \newcommand etc.)
stack=[]; envbad=[]
for kind,name in re.findall(r"\\(begin|end)\{([^}]*)\}", mains):
    if kind=="begin": stack.append(name)
    else:
        if not stack or stack[-1]!=name: envbad.append(f"{kind}{{{name}}} (stack top={stack[-1] if stack else None})")
        elif stack: stack.pop()
if stack: envbad.append("unclosed: "+", ".join(stack))
print(f"[env] balance issues: {len(envbad)}")
if envbad: err.append("env issues: "+"; ".join(envbad)); print("   ->", envbad)

# 6. brace balance (rough)
nb = mains.count("{")-mains.count("}")
print(f"[brace] net {{ - }} in main.tex = {nb} (0 expected, approximate)")

print("\n=== %d issue group(s) ===" % len(err))
sys.exit(1 if err else 0)
