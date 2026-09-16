#!/usr/bin/env python
from __future__ import annotations
import argparse,csv,re
from pathlib import Path

ACTIVE=('mc_physics.py','mc_pool_config.py','mc_parametric.py','mc_online_physics.py')
PATTERNS=[
    re.compile(r'\(\(s[^\n]*-\s*mass\)\s*\*\*\s*2'),
    re.compile(r'\(s\s*-\s*mass\)\.square\('),
    re.compile(r's_res\s*=\s*mass\s*\+\s*width'),
    re.compile(r'\(s_row\s*-\s*m\)\s*\*\*\s*2'),
    re.compile(r'\(\(s-m\)\^2'),
    re.compile(r'z=atan\(\(s-m\)/'),
]

def stale_hits(path:Path):
    text=path.read_text(encoding='utf-8',errors='ignore')
    rows=[]
    for lineno,line in enumerate(text.splitlines(),1):
        for pat in PATTERNS:
            if pat.search(line):
                rows.append((lineno,line.strip()))
                break
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--output',default='corrected_physics_reset_results/stale_formula_references.csv')
    args=ap.parse_args()
    root=Path(args.root)
    active_errors=[]; all_rows=[]
    for name in ACTIVE:
        path=root/name
        if not path.exists():
            active_errors.append(f'missing active physics file: {name}')
            continue
        hits=stale_hits(path)
        for line,text in hits:
            active_errors.append(f'{name}:{line}: {text}')
    for path in root.rglob('*.py'):
        if any(part in {'.git','venv','.venv','__pycache__'} for part in path.parts):
            continue
        for line,text in stale_hits(path):
            all_rows.append({'file':str(path.relative_to(root)),'line':line,'text':text,'active':int(path.name in ACTIVE)})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['file','line','text','active']); w.writeheader(); w.writerows(all_rows)
    print('active physics consistency:', 'PASS' if not active_errors else 'FAIL')
    print('legacy/stale references found:',len(all_rows))
    print('report:',out)
    if all_rows:
        print('NOTE: old experiment/validator files may still contain the wrong formula. Treat them as archived until patched.')
    if active_errors:
        print('\n'.join(active_errors)); raise SystemExit(2)

if __name__=='__main__':
    main()
