#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strict integrity + alias-numerics validation for Exp36-v2."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from mc_physics import scaled_forward_observation_numpy
from exp36_forward64 import forward64_batched


def write_csv(path,rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))


def load_manifest(path):
    by={'train':{},'val':{},'test':{}}
    for r in read_csv(path):
        by[r['split']][int(r['state_id'])]=np.array([float(r['a1']),float(r['a2']),float(r['a3']),float(r['m']),float(r['gamma'])],dtype=np.float64)
    return by


def margin(gtrue,galias,ref_noise):
    return float(np.sqrt(np.mean((galias-gtrue)**2))/max(ref_noise*np.sqrt(np.mean(gtrue*gtrue)),1e-30))


def unacceptable(true,alias,tol):
    da=abs(alias[0]-true[0]); dm=abs(alias[3]-true[3]); gf=max(alias[4]/true[4],true[4]/alias[4])
    eps=2e-10
    return (da+eps>=float(tol['a1_abs'])) or (dm+eps>=float(tol['m_abs'])) or (gf+eps>=float(tol['gamma_factor']))


def trigger_ok(trigger,true,alias,tol):
    eps=2e-10
    if trigger=='a1_low': return alias[0] <= true[0]-float(tol['a1_abs'])+eps
    if trigger=='a1_high': return alias[0] >= true[0]+float(tol['a1_abs'])-eps
    if trigger=='m_low': return alias[3] <= true[3]-float(tol['m_abs'])+eps
    if trigger=='m_high': return alias[3] >= true[3]+float(tol['m_abs'])-eps
    d=math.log10(float(tol['gamma_factor']))
    if trigger=='gamma_low': return math.log10(alias[4]) <= math.log10(true[4])-d+eps
    if trigger=='gamma_high': return math.log10(alias[4]) >= math.log10(true[4])+d-eps
    return False


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='data_exp36_3p'); ap.add_argument('--max-forward-states',type=int,default=0)
    args=ap.parse_args(); root=Path(args.data_dir)
    meta=json.loads((root/'metadata.json').read_text(encoding='utf-8')); manifest=load_manifest(root/'split_manifest.csv')
    integ=int(meta['physics_integration_points']); ref=float(meta['reference_noise']); tol=meta['recovery_tolerances']; pr=meta['parameter_ranges']
    numerics=meta.get('numerics',{}) if 'numerics' in meta else {}
    # Config numerical defaults are also encoded in metadata top-level summary fields.
    rtol=float(numerics.get('alias_margin_recompute_rtol',2e-6)); atol=float(numerics.get('alias_margin_recompute_atol',2e-9)); consistency_tol=float(numerics.get('grid_refined_consistency_margin_tol',1e-9))
    checks=[]; errors=[]; warnings=[]

    sets={k:set(v) for k,v in manifest.items()}
    for name,ov in [('train_val',sets['train']&sets['val']),('train_test',sets['train']&sets['test']),('val_test',sets['val']&sets['test'])]:
        ok=not ov; checks.append({'check':f'state_leakage_{name}','status':'PASS' if ok else 'FAIL','value':len(ov)})
        if not ok: errors.append(f'physical-state leakage {name}')

    required={'gy','g_clean','a1','a2','a3','m','gamma','state_id','noise_level'}; forward_states={}
    for ndir in sorted(root.glob('noise_*pct')):
        for sp in ('train','val','test'):
            path=ndir/f'{sp}.npz'
            if not path.exists(): errors.append(f'missing {path}'); continue
            with np.load(path,allow_pickle=False) as z:
                missing=required-set(z.files)
                if missing: errors.append(f'{path}: missing {sorted(missing)}'); continue
                n=len(z['state_id']); lengths=all(len(z[k])==n for k in required); finite=all(np.isfinite(z[k]).all() for k in ('gy','g_clean','a1','a2','a3','m','gamma','noise_level'))
                ids=z['state_id'].astype(np.int64); unique=set(map(int,np.unique(ids))); expected=set(manifest[sp]); ids_ok=unique==expected
                param_ok=True; clean_ok=True
                for sid in unique:
                    mask=ids==sid; exp=manifest[sp].get(sid)
                    obs=np.array([z['a1'][mask][0],z['a2'][mask][0],z['a3'][mask][0],z['m'][mask][0],z['gamma'][mask][0]],dtype=np.float64)
                    if exp is None or not np.allclose(obs,exp,rtol=0,atol=2e-7): param_ok=False; break
                    gc=z['g_clean'][mask]
                    if len(gc)>1 and not np.allclose(gc,gc[:1],rtol=0,atol=2e-7): clean_ok=False; break
                    forward_states.setdefault(sid,(exp.copy(),gc[0].astype(np.float64).copy()))
                tag=f'{ndir.name}_{sp}'
                for suf,ok,val in [('array_lengths',lengths,n),('finite',finite,int(finite)),('state_ids_match_manifest',ids_ok,len(unique)),('parameters_match_manifest',param_ok,int(param_ok)),('g_clean_constant_within_state',clean_ok,int(clean_ok))]:
                    checks.append({'check':f'{tag}_{suf}','status':'PASS' if ok else 'FAIL','value':val})
                    if not ok: errors.append(f'{tag}: {suf} failed')

    items=sorted(forward_states.items())
    if args.max_forward_states>0: items=items[:args.max_forward_states]
    if items:
        pars=np.stack([x[1][0] for x in items]); saved=np.stack([x[1][1] for x in items])
        recalc=scaled_forward_observation_numpy(pars,integration_points=integ).astype(np.float64)
        rel=float(np.linalg.norm(recalc-saved)/max(np.linalg.norm(saved),1e-30)); ok=rel<=1e-10
        checks.append({'check':'direct_forward_relL2','status':'PASS' if ok else 'FAIL','value':rel})
        if not ok: errors.append(f'direct forward mismatch {rel:g}')
        f64=forward64_batched(pars,integ); roundtrip=float(np.linalg.norm(f64.astype(np.float32).astype(np.float64)-recalc)/max(np.linalg.norm(recalc),1e-30)); ok2=roundtrip<=1e-7
        checks.append({'check':'forward64_roundtrip_relL2','status':'PASS' if ok2 else 'FAIL','value':roundtrip})
        if not ok2: errors.append(f'float64 forward mismatch {roundtrip:g}')

    # Alias diagnostic re-verification from saved parameters, independent of saved margins.
    alias_rows=read_csv(root/'profiled_alias_selected.csv')
    union={}; [union.update(x) for x in manifest.values()]
    max_refined_minus_grid=-math.inf; max_refined_margin_err=0.0; max_grid_margin_err=0.0; bad_alias=0; bad_trigger=0; bad_bounds=0
    for r in alias_rows:
        sid=int(r['state_id']); true=union[sid]
        grid=np.array([float(r['grid_alias_a1_verified']),true[1],true[2],float(r['grid_alias_m_verified']),float(r['grid_alias_gamma_verified'])],dtype=np.float64)
        refined=np.array([float(r['alias_a1_refined']),true[1],true[2],float(r['alias_m_refined']),float(r['alias_gamma_refined'])],dtype=np.float64)
        allp=np.stack([true,grid,refined]); gg=forward64_batched(allp,integ); mg=margin(gg[0],gg[1],ref); mr=margin(gg[0],gg[2],ref)
        sg=float(r['recovery_alias_margin_grid_verified']); sr=float(r['recovery_alias_margin_refined'])
        max_grid_margin_err=max(max_grid_margin_err,abs(mg-sg)); max_refined_margin_err=max(max_refined_margin_err,abs(mr-sr)); max_refined_minus_grid=max(max_refined_minus_grid,sr-sg)
        if not unacceptable(true,refined,tol): bad_alias+=1
        if not trigger_ok(r['alias_trigger_refined'],true,refined,tol): bad_trigger+=1
        if not (pr['a1'][0]-1e-12<=refined[0]<=pr['a1'][1]+1e-12 and pr['m'][0]-1e-12<=refined[3]<=pr['m'][1]+1e-12 and pr['gamma'][0]-1e-12<=refined[4]<=pr['gamma'][1]+1e-12): bad_bounds+=1
    grid_ok=max_grid_margin_err<=atol+rtol*max(1.0,max(float(r['recovery_alias_margin_grid_verified']) for r in alias_rows))
    refined_ok=max_refined_margin_err<=atol+rtol*max(1.0,max(float(r['recovery_alias_margin_refined']) for r in alias_rows))
    consistency=max_refined_minus_grid<=consistency_tol
    for name,ok,val in [('grid_alias_margin_direct_recompute',grid_ok,max_grid_margin_err),('refined_alias_margin_direct_recompute',refined_ok,max_refined_margin_err),('refined_le_verified_grid',consistency,max_refined_minus_grid),('refined_alias_unacceptable',bad_alias==0,bad_alias),('refined_trigger_constraint',bad_trigger==0,bad_trigger),('refined_alias_in_parameter_bounds',bad_bounds==0,bad_bounds)]:
        checks.append({'check':name,'status':'PASS' if ok else 'FAIL','value':val})
        if not ok: errors.append(f'{name} failed: {val}')

    frac=float(meta.get('refined_recovery_margin',{}).get('fraction_ge_1',np.nan)); checks.append({'check':'fraction_recovery_alias_margin_ge_1_noise_rms','status':'INFO','value':frac})
    if np.isfinite(frac) and frac<0.5: warnings.append('Fewer than 50% of selected states have recovery-aligned alias margin >= 1 reference-noise RMS. This is a physics/conditioning warning, not a numerical-integrity failure.')

    payload={'pass':not errors,'errors':errors,'warnings':warnings,'checked_unique_states':len(items),'alias_rows_checked':len(alias_rows),'alias_numerics_version':meta.get('alias_numerics_version','unknown')}
    (root/'postbuild_validation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8'); write_csv(root/'postbuild_validation.csv',checks)
    print('='*90); print('EXP36-v2 POST-BUILD VALIDATION:','PASS' if not errors else 'FAIL'); print('errors:',errors); print('warnings:',warnings); print('read:',root/'postbuild_validation.csv'); print('='*90)
    if errors: raise SystemExit(2)

if __name__=='__main__': main()
