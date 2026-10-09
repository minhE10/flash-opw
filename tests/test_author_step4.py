"""CPU mathematics and evidence mutation tests; no CUDA pass is claimed."""
from copy import deepcopy
import sys
from types import SimpleNamespace
import pytest
import torch

from scripts.author_step4_reference import (PROTOCOL,inventory,inputs,tensor_hashes,
    fixed_potentials,sample_plan,relative,cost_matrix,decide,cg_confirmed)
from scripts.check_author_step4 import matched_legacy
from scripts.audit_author_step4 import recompute
from scripts.author_flashsinkhorn_fp64 import direct_hvp
from flashsinkhorn import sinkhorn_dense,hessian_vector_product
from scripts import run_author_step4 as runner


def test_inventory_covers_each_benchmark_sweep_and_both_precisions():
    cases=inventory();assert len(cases)==111 and len({c['id'] for c in cases})==111
    forward={tuple(c['shape']) for c in cases if c['tier']=='benchmark_forward'}
    for n in (5000,10000,20000,30000,40000,50000):
        assert (n,n,64) in forward and (n,n,1024) in forward
    for d in (4,8,16,32,64,128,256,512,1024):assert (20000,20000,d) in forward
    hvp={tuple(c['shape']) for c in cases if c['tier']=='benchmark_hvp'}
    for n in (5000,6000,7000,8000,9000,10000,20000,30000,40000,50000):assert (n,n,64) in hvp
    for d in (4,8,16,32,64,128,256,512):assert (10000,10000,d) in hvp
    dense=[c for c in cases if c['tier']=='dense']
    assert {c['eps'] for c in dense}=={0.01,0.1,1.0}
    assert {c['weights'] for c in dense}=={'uniform','nonuniform'}
    assert {c['precision'] for c in dense}=={'ieee','tf32'}
    assert tensor_hashes(inputs(dense[0]))==tensor_hashes(inputs(dense[0]))


def test_legacy_boundary_adapter_matches_author_schedule_not_native(monkeypatch):
    # Substitute CPU equations for GPU launches to check adapter orchestration.
    def update(q,k,old,bias,logw,out,scale,symmetric,precision,bm,bn):
        out.copy_(logw-torch.logsumexp(scale*(q@k.T)+bias[None,:],1))
    def symmetric(x,y,u,v,la,lb,un,vn,scale,precision,bm,bn):
        update(x,y,u,v,la,un,scale,False,precision,bm,bn)
        update(y,x,v,u,lb,vn,scale,False,precision,bm,bn)
        un.copy_((un+u)*0.5);vn.copy_((vn+v)*0.5)
    monkeypatch.setitem(sys.modules,'flashsinkhorn.triton_kernels',SimpleNamespace(update=update,symmetric_update=symmetric))
    case=dict(inventory()[0],eps=0.7,cost_scale=0.5,precision='ieee')
    x,y,a,b,_=[t.double() for t in inputs(case)]
    result=matched_legacy(x,y,a,b,case,3)
    f,g=fixed_potentials(x,y,a,b,0.7,0.5,3,'symmetric',True)
    assert result.n_iters==5
    torch.testing.assert_close(result.f,f,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(result.g,g,rtol=1e-12,atol=1e-12)
    native=sinkhorn_dense(x,y,a=a,b=b,epsilon=0.7,cost_scale=0.5,n_iters=3,schedule='symmetric')
    nf,ng=fixed_potentials(x,y,a,b,0.7,0.5,3,'symmetric',False)
    torch.testing.assert_close(native.f,nf,rtol=1e-12,atol=1e-12)
    assert relative(f,nf)>1e-4


@pytest.mark.parametrize('eps',[0.1,0.7,1.0])
def test_hvp_damping_units_match_independent_kkt(eps):
    gen=torch.Generator().manual_seed(8)
    x,y=torch.randn(7,3,generator=gen,dtype=torch.float64)*0.2,torch.randn(9,3,generator=gen,dtype=torch.float64)*0.2
    r=sinkhorn_dense(x,y,epsilon=eps,n_iters=300)
    from flashsinkhorn import materialize_plan
    p=materialize_plan(r)
    v=torch.randn(x.shape,generator=gen,dtype=x.dtype)
    tau2=0.2  # Visible damping difference to detect a unit mismatch.
    hd,residual=direct_hvp(x,y,p,v,eps,tau2=tau2)
    actual=hessian_vector_product(r,v,damping=eps*tau2,max_cg_iters=256,cg_rtol=1e-12,cg_atol=1e-12)
    assert residual<1e-12
    torch.testing.assert_close(actual,hd,rtol=1e-8,atol=1e-10)
    if eps!=1.0:
        wrong=hessian_vector_product(r,v,damping=tau2,max_cg_iters=256,cg_rtol=1e-12,cg_atol=1e-12)
        assert relative(wrong,hd)>1e-5


def test_convergence_and_failure_accounting():
    checks={'operator':{'passed':True}}
    assert decide(checks,[])=='passed_checks'
    assert decide(checks,['CG cap'])=='passed_checks_with_coverage_gaps'
    assert decide({'operator':{'passed':False}},['CG cap'])=='failed'
    assert decide({},[])=='failed'
    assert not cg_confirmed({'cg_converged':False,'cg_residual':1e-20,'cg_initial_residual':1.0})
    assert not cg_confirmed({'cg_converged':True,'cg_residual':1e-3,'cg_initial_residual':1.0})


def synthetic_case():
    # Saved synthetic outputs are exact CPU equations, not simulated GPU evidence.
    case=inventory()[0];x,y,a,b,v=[t.double() for t in inputs(case)]
    n,m=len(x),len(y);rows,cols=torch.arange(n),torch.arange(m)
    eps,cs=case['eps'],case['cost_scale'];limit=PROTOCOL['ieee_relative_limit']
    arrays={'rows':rows,'cols':cols}
    for prefix,budget,author in [('fixed_author',10,True),('fixed_matched',10,True),
        ('fixed_native',10,False),('fixed_reference',10,True),('native_reference',10,False),
        ('operator',500,True),('solve_matched',500,True),('solve_reference',500,True)]:
        f,g=fixed_potentials(x,y,a,b,eps,cs,budget,'symmetric',author)
        arrays[prefix+'_f'],arrays[prefix+'_g']=f,g
    def pair(prefix):return sample_plan(x,y,a,b,arrays[prefix+'_f'],arrays[prefix+'_g'],eps,cs,rows,cols)
    p,pc=pair('operator');fp,fn=pair('fixed_author')[0],pair('fixed_native')[0]
    residual=max(float((p.sum(1)-a).abs().sum()),float((pc.sum(0)-b).abs().sum()))
    confirmed=residual<=PROTOCOL['reference_marginal_limit']
    gaps=[] if confirmed else ['Forward marginal not confirmed at fixed solve budget']
    r={'case':case,'protocol':PROTOCOL,'input_sha256':tensor_hashes(inputs(case)),
        'operator_iterations':500,'operator_updates':502,'updates':{'author':12,'matched_legacy':12,'native_legacy':10},
        'convergence':{'confirmed':confirmed,'author_marginal_l1':residual,'reference_marginal_l1':residual},
        'diagnostics':{'native_schedule_plan_relative_l2':relative(fp,fn)},'coverage_gaps':gaps,'checks':{}}
    for axis,pp,keys in [(1,p,y),(0,pc.T,x)]:
        for backend in ('author','legacy'):
            arrays[f'{backend}_apply_{axis}']=pp@(keys-0.5)
            arrays[f'{backend}_mass_{axis}']=pp.sum(1)
    for side,pp,points,keys,weights in [('x',p,x,y,a),('y',pc.T,y,x,b)]:
        mass=pp.sum(1);mean=(pp@keys)/mass[:,None]
        normalized=2*cs*weights[:,None]*(points-mean)
        actual=2*cs*(mass[:,None]*points-pp@keys)
        arrays['author_gradient_'+side]=normalized;arrays['legacy_gradient_'+side]=actual
        r['diagnostics']['gradient_convention_difference_'+side]=relative(normalized,actual)
    fo,go=arrays['operator_f']+eps*a.log(),arrays['operator_g']+eps*b.log()
    arrays['hvp_ott_f'],arrays['hvp_ott_g']=fo,go
    hp=torch.exp((fo[:,None]+go[None,:]-cost_matrix(x,y,cs))/eps)
    hd,_=direct_hvp(x,y,hp,v,eps,cs,1e-5)
    arrays.update(author_hvp=hd,legacy_hvp=hd,direct_hvp=hd)
    info={'cg_converged':True,'cg_residual':0.0,'cg_initial_residual':1.0,'cg_iters':1}
    r['hvp']={'author_cg':info,'legacy_cg':dict(info,damping=eps*1e-5),
        'cap':256,'tau2':1e-5,'legacy_schur_damping':eps*1e-5}
    names=[]
    for prefix in ('fixed_author_vs_matched_legacy','fixed_author_vs_fp64','fixed_matched_legacy_vs_fp64',
                   'native_legacy_vs_own_schedule_fp64','solve_author_vs_matched_legacy','solve_author_vs_fp64'):
        names.extend([prefix+'_rows',prefix+'_cols'])
    names.extend(f'{backend}_{kind}_{axis}' for axis in (1,0) for backend in ('author','legacy') for kind in ('apply','mass'))
    names.extend(['author_normalized_gradient_x','author_normalized_gradient_y','legacy_actual_mass_gradient_x','legacy_actual_mass_gradient_y'])
    for name in names:r['checks'][name]={'relative_l2':0.0,'limit':limit,'passed':True}
    for name in ('hvp_author_vs_legacy','hvp_author_vs_direct','hvp_legacy_vs_direct'):
        r['checks'][name]={'relative_l2':0.0,'limit':PROTOCOL['hvp_relative_limit'],'passed':True}
    r['status']=decide(r['checks'],gaps)
    return r,arrays


def test_audit_recomputes_numerical_outputs_and_rejects_tampering():
    r,arrays=synthetic_case()
    assert recompute(r,arrays)['status']==r['status']
    bad=dict(arrays,author_apply_1=arrays['author_apply_1']+0.1)
    with pytest.raises(AssertionError):recompute(r,bad)
    bad_record=deepcopy(r);bad_record['checks']['author_apply_1']['limit']=0.5
    with pytest.raises(AssertionError):recompute(bad_record,arrays)
    bad_record=deepcopy(r);bad_record['hvp']['legacy_schur_damping']=1e-5
    with pytest.raises(AssertionError):recompute(bad_record,arrays)


def test_missing_normal_direction_never_certifies_direct_hvp():
    r,arrays=synthetic_case()
    r['input_sha256']['direction']='different-platform-normal-output'
    with pytest.raises(AssertionError):recompute(r,arrays)
    incomplete=dict(arrays,direct_hvp=arrays['direct_hvp']+100)
    result=recompute(r,incomplete,allow_missing_direction=True)
    assert result['direction_verified'] is False
    assert result['direct_hvp_recomputed'] is False
    assert set(result['unverified_checks'])=={'hvp_author_vs_direct','hvp_legacy_vs_direct'}
    r['input_sha256']['x']='different-coordinates'
    with pytest.raises(AssertionError):recompute(r,arrays,allow_missing_direction=True)


def test_exact_supplement_restores_direct_hvp_audit(monkeypatch):
    from scripts import audit_author_step4 as auditor
    record,arrays=synthetic_case()
    original=inputs(record['case'])
    monkeypatch.setattr(auditor,'inputs',lambda case:(*original[:-1],original[-1]+0.01))
    with pytest.raises(AssertionError):auditor.recompute(record,arrays)
    result=auditor.recompute(record,arrays,direction=original[-1])
    assert result['direction_verified'] and result['direct_hvp_recomputed']
    assert result['unverified_checks']==[] and result['status']==record['status']


def test_direction_export_parts_bind_to_original_hashes(tmp_path,monkeypatch):
    import hashlib
    import json
    import zipfile
    from scripts import export_author_step4_directions as exporter
    cases=inventory()[:2]
    monkeypatch.setattr(exporter,'inventory',lambda:cases)
    monkeypatch.setattr(exporter,'MAX_PART_BYTES',256)
    helper=exporter.Path(exporter.__file__).with_name('author_step4_reference.py')
    summary={'inventory':cases,'runs':[{'id':c['id']} for c in cases],
        'validation_code_sha256':{helper.name:hashlib.sha256(helper.read_bytes()).hexdigest()}}
    members={'step4-summary.json':json.dumps(summary).encode()}
    for case in cases:
        members[case['id']+'/result.json']=json.dumps({'case':case,'hvp':{'present':True},
            'input_sha256':tensor_hashes(inputs(case))}).encode()
    members['artifact-sha256.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in members.items()}).encode()
    archive=tmp_path/'original.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for n,b in members.items():z.writestr(n,b)
    parts=exporter.export(archive,tmp_path/'exports')
    assert len(parts)==2
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    recovered=exporter.load_directions(parts,sha)
    for case in cases:
        torch.testing.assert_close(recovered[case['id']],inputs(case)[-1],rtol=0,atol=0)
    with pytest.raises(AssertionError):exporter.load_directions(parts[:1],sha)
    with pytest.raises(AssertionError):exporter.load_directions(parts,'wrong-original-archive')


def test_pinned_mat5_pruning_reproduces_invalid_keyword_dimension():
    # Characterize the upstream defect without importing Triton or changing
    # protected source. This is not a test that Mat5 is numerically correct.
    import ast
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'flash_sinkhorn_author/torch-ext/flash_sinkhorn/kernels/apply_ott.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mat5_prune_configs')
    namespace={}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    configs=[SimpleNamespace(kwargs={'BLOCK_D':d}) for d in (16,32,64,128,256,512,1024)]
    prune=namespace[function.name]
    # Triton 3.6 keeps positional named_args separate from launch kwargs.
    assert [c.kwargs['BLOCK_D'] for c in prune(configs,{},D=512)]==[64,128,256]
    assert all(c.kwargs['BLOCK_D']>=512 for c in prune(configs,{'D':512}))
    hvp_source=source.parents[1]/'hvp.py'
    calls=[n for n in ast.walk(ast.parse(hvp_source.read_text(encoding='utf-8')))
           if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='mat5_sqeuclid']
    assert len(calls)==1 and 'autotune' not in {k.arg for k in calls[0].keywords}


def test_parent_preserves_failure_continues_and_bundles(tmp_path,monkeypatch):
    import json
    import subprocess
    from pathlib import Path
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    monkeypatch.setattr(runner,'inventory',lambda:inventory()[:2])
    monkeypatch.setattr(runner,'verify',lambda:{'status':'verified'})
    monkeypatch.setattr(runner.metadata,'version',lambda name:'0.3.1' if name=='geomloss' else 'synthetic')
    monkeypatch.setattr(runner,'prepare_profile',lambda p:(p/'implementation-rtx5080',{}))
    monkeypatch.setattr(runner,'verify_profile',lambda p:{'status':'profile_verified'})
    (tmp_path/'scripts').mkdir();(tmp_path/'flashsinkhorn').mkdir()
    for name in ('author_step4_reference.py','check_author_step4.py','run_author_step4.py','author_flashsinkhorn_fp64.py'):
        (tmp_path/'scripts'/name).write_text('# simulated orchestration fixture')
    (tmp_path/'flashsinkhorn'/'solver.py').write_text('# simulated legacy source')
    monkeypatch.setattr(runner.subprocess,'run',lambda command,**kwargs:
        subprocess.CompletedProcess(command,0,stdout='synthetic',stderr=''))
    class Process:
        def __init__(self,command,**kwargs):
            self.pid,self.stdout=7,iter([])
            name=command[command.index('--case-id')+1]
            self.code=int(name=='case_000')
            target=Path(command[command.index('--output')+1]);target.mkdir()
            (target/'result.json').write_text(json.dumps({'status':'failed' if self.code else 'passed_checks_with_coverage_gaps'}))
        def wait(self):return self.code
        def poll(self):return self.code
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    assert runner.main()==1
    summary_path=next((tmp_path/'outputs').glob('*/step4-summary.json'))
    result=json.loads(summary_path.read_text())
    assert len(result['runs'])==2 and result['missing_cases']==[]
    assert result['status']=='failed' and result['failed_cases']==['case_000']
    assert result['legacy_before']==result['legacy_after']
    assert summary_path.parent.with_suffix('.zip').exists()


def test_zip_audit_checks_complete_evidence_and_checksum(tmp_path,monkeypatch):
    # CPU-only synthetic evidence tests the transfer format, not GPU correctness.
    import hashlib
    from io import BytesIO
    import json
    import zipfile
    import numpy as np
    from scripts import audit_author_step4 as auditor
    record,arrays=synthetic_case()
    environment={'torch':'synthetic','triton':'synthetic','numpy':'synthetic'}
    record['environment']=environment
    monkeypatch.setattr(auditor,'inventory',lambda:[record['case']])
    pin=auditor.load_manifest()
    source={'status':'verified','commit':pin['commit'],'files_checked':131,'errors':[]}
    legacy={p.relative_to(auditor.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((auditor.ROOT/'flashsinkhorn').rglob('*.py'))}
    helpers=('author_step4_reference.py','check_author_step4.py','run_author_step4.py','author_flashsinkhorn_fp64.py')
    summary={'inventory':[record['case']],'protocol':PROTOCOL,'source':source,'source_after':source,
        'profile_after':{'status':'profile_verified','files_checked':131,'errors':[]},
        'legacy_before':legacy,'legacy_after':legacy,'environment':environment,
        'validation_code_sha256':{n:hashlib.sha256((auditor.ROOT/'scripts'/n).read_bytes()).hexdigest() for n in helpers},
        'runs':[{'id':record['case']['id'],'status':record['status'],'exit_code':0}],
        'status':record['status'],'missing_cases':[],'failed_cases':[],
        'coverage_gaps':[record['case']['id']] if record['coverage_gaps'] else []}
    buffer=BytesIO();np.savez_compressed(buffer,**{n:t.numpy() for n,t in arrays.items()})
    members={'step4-summary.json':json.dumps(summary).encode(),
        'profile/kernel-profile.json':json.dumps({'files':auditor.expected_entries(),'upstream_commit':pin['commit']}).encode(),
        'case_000/result.json':json.dumps(record).encode(),'case_000/arrays.npz':buffer.getvalue()}
    members['artifact-sha256.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in members.items()}).encode()
    archive=tmp_path/'cpu-format-fixture.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name,data in members.items():z.writestr(name,data)
    assert auditor.audit(archive)['status']==record['status']
    members['case_000/result.json']+=b' '
    with zipfile.ZipFile(archive,'w') as z:
        for name,data in members.items():z.writestr(name,data)
    with pytest.raises(AssertionError):auditor.audit(archive)
