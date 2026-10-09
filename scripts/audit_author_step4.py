"""Audit saved step-4 arrays with CPU FP64; never execute archive contents."""
import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import zipfile
import numpy as np
import torch

try:
    from .author_step4_reference import (PROTOCOL,inventory,inputs,tensor_hashes,selected,
        fixed_potentials,sample_plan,cost_matrix,relative,check,cg_confirmed,decide)
    from .author_flashsinkhorn_sources import ROOT,load_manifest
    from .author_flashsinkhorn_profiles import expected_entries
    from .author_flashsinkhorn_fp64 import direct_hvp
except ImportError:
    from author_step4_reference import (PROTOCOL,inventory,inputs,tensor_hashes,selected,
        fixed_potentials,sample_plan,cost_matrix,relative,check,cg_confirmed,decide)
    from author_flashsinkhorn_sources import ROOT,load_manifest
    from author_flashsinkhorn_profiles import expected_entries
    from author_flashsinkhorn_fp64 import direct_hvp


def recompute(record,arrays):
    case=record['case'];dense=case['tier']=='dense'
    assert record['protocol']==PROTOCOL
    raw=inputs(case);assert tensor_hashes(raw)==record['input_sha256'], 'Regenerated input hashes differ'
    x,y,a,b,v=[t.double() for t in raw]
    rows,cols=selected(len(x),dense),selected(len(y),dense)
    assert torch.equal(rows,arrays['rows']) and torch.equal(cols,arrays['cols'])
    if record.get('error'):
        assert record['status']=='failed'
        return {'status':'failed','error':record['error'],'partial_evidence':True}
    checks={};gaps=[]
    limit=PROTOCOL[case['precision']+'_relative_limit'];eps,cs=case['eps'],case['cost_scale']
    def pair(prefix):
        return sample_plan(x,y,a,b,arrays[prefix+'_f'].double(),arrays[prefix+'_g'].double(),eps,cs,rows,cols)
    def compare(name,actual,expected,threshold=limit):
        checks[name]=check(actual.double(),expected.double(),threshold)
    def pairs(name,first,second):
        compare(name+'_rows',first[0],second[0]);compare(name+'_cols',first[1],second[1])
    pa,pl,pn=pair('fixed_author'),pair('fixed_matched'),pair('fixed_native')
    pairs('fixed_author_vs_matched_legacy',pa,pl)
    assert math.isclose(relative(pa[0],pn[0]),record['diagnostics']['native_schedule_plan_relative_l2'],rel_tol=1e-7,abs_tol=1e-12)
    assert record['updates']=={'author':10+(2 if case['mode']=='symmetric' else 0),
        'matched_legacy':10+(2 if case['mode']=='symmetric' else 0),'native_legacy':10}
    if dense:
        for prefix,native in [('fixed_reference',True),('native_reference',False)]:
            ff,gg=fixed_potentials(x,y,a,b,eps,cs,10,case['mode'],native)
            assert relative(arrays[prefix+'_f'].double(),ff)<1e-10
            assert relative(arrays[prefix+'_g'].double(),gg)<1e-10
        pr,pnr=pair('fixed_reference'),pair('native_reference')
        pairs('fixed_author_vs_fp64',pa,pr);pairs('fixed_matched_legacy_vs_fp64',pl,pr)
        pairs('native_legacy_vs_own_schedule_fp64',pn,pnr)
    iterations=500 if dense else 100 if case['tier']=='benchmark_hvp' else 10
    assert record['operator_iterations']==iterations
    assert record['operator_updates']==iterations+(2 if case['mode']=='symmetric' else 0)
    p_rows,p_cols=pair('operator')
    if iterations!=10:
        pairs('solve_author_vs_matched_legacy',(p_rows,p_cols),pair('solve_matched'))
    if dense:
        ff,gg=fixed_potentials(x,y,a,b,eps,cs,iterations,case['mode'])
        assert relative(arrays['solve_reference_f'].double(),ff)<1e-10
        assert relative(arrays['solve_reference_g'].double(),gg)<1e-10
        pr=pair('solve_reference');pairs('solve_author_vs_fp64',(p_rows,p_cols),pr)
        residual=max(float((p_rows.sum(1)-a).abs().sum()),float((p_cols.sum(0)-b).abs().sum()))
        rr=max(float((pr[0].sum(1)-a).abs().sum()),float((pr[1].sum(0)-b).abs().sum()))
        confirmed=residual<=PROTOCOL['marginal_limit'] and rr<=PROTOCOL['reference_marginal_limit']
        assert record['convergence']['confirmed']==confirmed
        assert math.isclose(residual,record['convergence']['author_marginal_l1'],rel_tol=1e-7,abs_tol=1e-12)
        assert math.isclose(rr,record['convergence']['reference_marginal_l1'],rel_tol=1e-7,abs_tol=1e-12)
        if not confirmed:gaps.append('Forward marginal not confirmed at fixed solve budget')
    else:
        assert record['convergence']['kind']=='sampled_rows_and_columns_only' and record['convergence']['confirmed'] is False
        gaps.append('Large-shape checks sample rows/columns; no full marginal certificate or full FP64 solve')
    for axis,p,keys in [(1,p_rows,y),(0,p_cols.T,x)]:
        expected=p@(keys-0.5);mass=p.sum(1)
        for backend in ('author','legacy'):
            compare(f'{backend}_apply_{axis}',arrays[f'{backend}_apply_{axis}'],expected)
            compare(f'{backend}_mass_{axis}',arrays[f'{backend}_mass_{axis}'],mass)
    for side,p,points,keys,weights,indices in [('x',p_rows,x,y,a,rows),('y',p_cols.T,y,x,b,cols)]:
        mass=p.sum(1);mean=(p@keys)/mass[:,None]
        normalized=2*cs*weights[indices,None]*(points[indices]-mean)
        actual_mass=2*cs*(mass[:,None]*points[indices]-p@keys)
        compare('author_normalized_gradient_'+side,arrays['author_gradient_'+side],normalized)
        compare('legacy_actual_mass_gradient_'+side,arrays['legacy_gradient_'+side],actual_mass)
        assert math.isclose(relative(normalized,actual_mass),record['diagnostics']['gradient_convention_difference_'+side],rel_tol=1e-7,abs_tol=1e-12)
    if dense or case['tier']=='benchmark_hvp':
        h=record['hvp'];cap=256 if dense else 50
        assert h['cap']==cap and h['tau2']==1e-5 and h['legacy_schur_damping']==eps*1e-5
        assert h['legacy_cg']['damping']==eps*1e-5
        for info in (h['author_cg'],h['legacy_cg']):
            assert 0<=info['cg_iters']<=cap
        ha,hl=arrays['author_hvp'],arrays['legacy_hvp']
        assert torch.isfinite(ha).all() and torch.isfinite(hl).all()
        confirmed=cg_confirmed(h['author_cg']) and cg_confirmed(h['legacy_cg'])
        if confirmed:compare('hvp_author_vs_legacy',ha,hl,PROTOCOL['hvp_relative_limit'])
        else:gaps.append('HVP CG not confirmed in both backends; parity metric is diagnostic only')
        if dense:
            fo,go=arrays['hvp_ott_f'].double(),arrays['hvp_ott_g'].double()
            hp=torch.exp((fo[:,None]+go[None,:]-cost_matrix(x,y,cs))/eps)
            hd,lr=direct_hvp(x,y,hp,v,eps,cs,1e-5)
            assert lr<=PROTOCOL['linear_residual_limit']
            assert relative(arrays['direct_hvp'].double(),hd)<1e-8
            if confirmed:
                compare('hvp_author_vs_direct',ha,hd,PROTOCOL['hvp_relative_limit'])
                compare('hvp_legacy_vs_direct',hl,hd,PROTOCOL['hvp_relative_limit'])
    assert set(checks)==set(record['checks']), 'Missing or extra numerical checks'
    for name,result in checks.items():
        saved=record['checks'][name]
        assert saved['limit']==result['limit'] and saved['passed']==result['passed'],name
        assert math.isclose(saved['relative_l2'],result['relative_l2'],rel_tol=1e-7,abs_tol=1e-12),name
    assert gaps==record['coverage_gaps']
    status=decide(checks,gaps)
    assert status==record['status']
    return {'status':status,'failed_checks':[name for name,c in checks.items() if not c['passed']],
            'coverage_gaps':gaps}


def audit(archive):
    torch.set_num_threads(2)
    errors=[];results=[];pin=load_manifest()
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert len(names)==len(set(names)) and sum(i.file_size for i in z.infolist())<=512*2**20
        read=lambda n:json.loads(z.read(n))
        hashes=read('artifact-sha256.json')
        assert set(hashes)==set(names)-{'artifact-sha256.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hashes.items())
        s=read('step4-summary.json');assert s['inventory']==inventory() and s['protocol']==PROTOCOL
        for key in ('source','source_after'):
            assert s[key]['status']=='verified' and s[key]['commit']==pin['commit']
            assert s[key]['files_checked']==131 and s[key]['errors']==[]
        assert s['profile_after']=={'status':'profile_verified','files_checked':131,'errors':[]}
        profile=read('profile/kernel-profile.json')
        assert profile['files']==expected_entries() and profile['upstream_commit']==pin['commit']
        legacy={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT/'flashsinkhorn').rglob('*.py'))}
        assert s['legacy_before']==s['legacy_after']==legacy
        expected_helpers={'author_step4_reference.py','check_author_step4.py','run_author_step4.py','author_flashsinkhorn_fp64.py'}
        assert set(s['validation_code_sha256'])==expected_helpers
        for n,h in s['validation_code_sha256'].items():
            assert hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest()==h,'Validator version differs from uploaded run'
        assert [r['id'] for r in s['runs']]==[c['id'] for c in inventory()] and not s['missing_cases']
        environments=[]
        for run,case in zip(s['runs'],inventory()):
            print(f"[audit-step4] {case['id']}",flush=True)
            record=read(case['id']+'/result.json');assert record['case']==case
            environments.append(record['environment'])
            for key in ('torch','triton','numpy'):
                assert record['environment'][key]==s['environment'][key]
            # Bound nested NPZ expansion and forbid object/pickle data.
            raw=z.read(case['id']+'/arrays.npz')
            with zipfile.ZipFile(BytesIO(raw)) as nested:
                assert sum(i.file_size for i in nested.infolist())<=128*2**20
                assert len(nested.namelist())==len(set(nested.namelist()))
            with np.load(BytesIO(raw),allow_pickle=False) as saved:
                arrays={n:torch.from_numpy(saved[n].copy()) for n in saved.files}
            try:
                result=recompute(record,arrays)
                assert result['status']==run['status']
                assert (run['exit_code']==0)==(result['status']!='failed')
                results.append(dict(id=case['id'],**result))
            except Exception as exc:
                errors.append(f"{case['id']}: {type(exc).__name__}: {exc}")
        if any(e!=environments[0] for e in environments):errors.append('Case environments differ')
        assert s['failed_cases']==[r['id'] for r in s['runs'] if r['exit_code'] or r['status']=='failed']
        assert s['coverage_gaps']==[r['id'] for r in s['runs'] if r['status']=='passed_checks_with_coverage_gaps']
        status='failed' if any(r['status']=='failed' for r in results) else 'passed_checks_with_coverage_gaps' if any(r['status'].endswith('coverage_gaps') for r in results) else 'passed_checks'
        if status!=s['status']:errors.append('Aggregate status mismatch')
    return {'status':'failed_audit' if errors else status,'cases':results,'errors':errors,
        'limitation':'CPU FP64 audit of saved outputs and regenerated hash-matched inputs; no CUDA rerun or server authentication.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive',type=Path)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    try:result=audit(args.archive)
    except Exception as exc:result={'status':'failed_audit','errors':[f'{type(exc).__name__}: {exc}']}
    print(json.dumps(result,indent=2))
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return 0 if result['status']=='passed_checks' else 3 if result['status']=='passed_checks_with_coverage_gaps' else 1


if __name__=='__main__':raise SystemExit(main())
