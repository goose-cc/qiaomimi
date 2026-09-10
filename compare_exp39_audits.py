#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rank completed Exp39 deep audits without pretending the rank is a 95%-recovery guarantee."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path


def read_one(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return next(csv.DictReader(f))

def num(r,k):return float(r[k])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit-dir',default='exp39_targeted_audit');args=ap.parse_args();root=Path(args.audit_dir);rows=[]
    for p in sorted(root.glob('*_audit_summary.csv')):
        r=read_one(p);rows.append(r)
    if not rows:raise SystemExit(f'No *_audit_summary.csv in {root}')
    rows.sort(key=lambda r:(num(r,'alias_mahalanobis_p10'),num(r,'worst_target_slice_alias_mahalanobis_p10'),-num(r,'jac_condition_3p_median')),reverse=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    fields=['deep_rank']+fields
    out=[]
    for i,r in enumerate(rows,1):rr={'deep_rank':i,**r};out.append(rr)
    with (root/'exp39_deep_audit_ranking.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
    winner=out[0]['design_id'];(root/'exp39_current_winner.json').write_text(json.dumps({'design_id':winner,'note':'Current winner by continuous 3P P10, worst target-slice P10, then condition number. This is not yet a 95% recovery claim; next step is full 48k continuous screening + VarPro ceiling.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Current deep-audit winner:',winner);print('Read:',root/'exp39_deep_audit_ranking.csv')
if __name__=='__main__':main()
