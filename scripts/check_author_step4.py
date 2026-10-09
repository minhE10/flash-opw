"""One isolated GPU case: native author, matched legacy, CPU FP64 operators."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import traceback
import warnings
import numpy as np
import torch

try:
    from .author_step4_reference import (PROTOCOL,inventory,inputs,tensor_hashes,selected,
        fixed_potentials,sample_plan,cost_matrix,relative,check,cg_confirmed,decide)
    from .author_flashsinkhorn_fp64 import direct_hvp
except ImportError:
    from author_step4_reference import (PROTOCOL,inventory,inputs,tensor_hashes,selected,
        fixed_potentials,sample_plan,cost_matrix,relative,check,cg_confirmed,decide)
    from author_flashsinkhorn_fp64 import direct_hvp


def matched_legacy(x,y,a,b,case,iterations):
    """Original legacy kernels, with the author's explicit symmetric boundaries.

    This is a validation adapter, not a modification of either source tree.
    The native legacy high-level solver is measured separately below.
    """
    from flashsinkhorn import sinkhorn_flash, SinkhornResult
    if case['mode']=='alternating':
        return sinkhorn_flash(x,y,a=a,b=b,epsilon=case['eps'],cost_scale=case['cost_scale'],
            n_iters=iterations,schedule='alternating',tol=None,precision=case['precision'])
    from flashsinkhorn.triton_kernels import update,symmetric_update
    eps,cs=case['eps'],case['cost_scale']
    la,lb=a.log(),b.log()
    u,v=la-cs*x.square().sum(1)/eps,lb-cs*y.square().sum(1)/eps
    scale=2*cs/eps
    def full(u,v):
        un,vn=torch.empty_like(u),torch.empty_like(v)
        update(x,y,u,v,la,un,scale,False,case['precision'],32,64)
        update(y,x,v,u,lb,vn,scale,False,case['precision'],32,64)
        return un,vn
    u,v=full(u,v)
    for _ in range(iterations):
        un,vn=torch.empty_like(u),torch.empty_like(v)
        symmetric_update(x,y,u,v,la,lb,un,vn,scale,case['precision'],32,64)
        u,v=un,vn
    u,v=full(u,v)
    return SinkhornResult(x,y,a,b,u,v,eps,cs,iterations+2,'triton',case['precision'],32,64)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation',type=Path,required=True)
    parser.add_argument('--case-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    case=next(c for c in inventory() if c['id']==args.case_id)
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    sys.path.insert(0,str(args.implementation.resolve()/'torch-ext'))
    import flash_sinkhorn,flashsinkhorn
    from flash_sinkhorn.sinkhorn_solvers import sinkhorn_flashstyle_symmetric,sinkhorn_flashstyle_alternating
    from flash_sinkhorn.kernels.apply_flash import apply_plan_mat_flashstyle,apply_plan_vec_flashstyle
    from flash_sinkhorn.kernels.sinkhorn_triton_grad_sqeuclid import sinkhorn_geomloss_online_grad_sqeuclid
    from flash_sinkhorn.hvp import hvp_x_sqeuclid_from_potentials
    from flashsinkhorn import sinkhorn_flash,SinkhornResult,apply_plan,source_gradient,target_gradient,hessian_vector_product
    assert args.implementation.resolve() in Path(flash_sinkhorn.__file__).resolve().parents
    assert root/'flashsinkhorn' in Path(flashsinkhorn.__file__).resolve().parents
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.45)
    torch.backends.cuda.matmul.allow_tf32=False  # Torch reference-side math, independent of explicit Triton flags.
    torch.set_float32_matmul_precision('highest')
    report={'case':case,'protocol':PROTOCOL,'status':'running','checks':{},'coverage_gaps':[],
        'author_package':str(flash_sinkhorn.__file__),'legacy_package':str(flashsinkhorn.__file__),
        'warnings':[],'convergence':{},'diagnostics':{}}
    import triton
    report['environment']={'python':sys.version,'torch':torch.__version__,'triton':triton.__version__,
        'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda}
    arrays={}
    def save_array(name,t):
        arrays[name]=t.detach().cpu().contiguous().numpy().copy()
    def compare(name,actual,expected,limit):
        report['checks'][name]=check(actual.double(),expected.double(),limit)
    data=inputs(case)
    report['input_sha256']=tensor_hashes(data)
    xd,yd,ad,bd,vd=[t.double() for t in data]
    x,y,a,b,v=[t.cuda() for t in data]
    dense=case['tier']=='dense'
    rows,cols=selected(len(x),dense),selected(len(y),dense)
    save_array('rows',rows);save_array('cols',cols)
    eps,cs=case['eps'],case['cost_scale']
    limit=PROTOCOL[case['precision']+'_relative_limit']
    def author(iterations):
        kwargs=dict(eps=eps,n_iters=iterations,cost_scale=cs,allow_tf32=case['precision']=='tf32',
                    use_exp2=True,autotune=False,threshold=None,return_n_iters=True)
        if case['mode']=='symmetric':
            kwargs.update(use_epsilon_scaling=False,last_extrapolation=True)
            return sinkhorn_flashstyle_symmetric(x,y,a,b,**kwargs)
        return sinkhorn_flashstyle_alternating(x,y,a,b,**kwargs)
    def samples(f,g):
        return sample_plan(xd,yd,ad,bd,f.double().cpu(),g.double().cpu(),eps,cs,rows,cols)
    def plan_check(name,first,second):
        compare(name+'_rows',first[0],second[0],limit)
        compare(name+'_cols',first[1],second[1],limit)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            f,g,used=author(PROTOCOL['fixed_iterations'])
            old=matched_legacy(x,y,a,b,case,PROTOCOL['fixed_iterations'])
            native=sinkhorn_flash(x,y,a=a,b=b,epsilon=eps,cost_scale=cs,
                n_iters=PROTOCOL['fixed_iterations'],schedule=case['mode'],tol=None,precision=case['precision'])
            assert used==old.n_iters==PROTOCOL['fixed_iterations']+(2 if case['mode']=='symmetric' else 0)
            report['updates']={'author':used,'matched_legacy':old.n_iters,'native_legacy':native.n_iters}
            for prefix,ff,gg in [('fixed_author',f,g),('fixed_matched',old.f,old.g),('fixed_native',native.f,native.g)]:
                save_array(prefix+'_f',ff);save_array(prefix+'_g',gg)
            pa,pl,pn=samples(f,g),samples(old.f,old.g),samples(native.f,native.g)
            plan_check('fixed_author_vs_matched_legacy',pa,pl)
            report['diagnostics']['native_schedule_plan_relative_l2']=relative(pa[0],pn[0])
            if dense:
                rf,rg=fixed_potentials(xd,yd,ad,bd,eps,cs,PROTOCOL['fixed_iterations'],case['mode'])
                nf,ng=fixed_potentials(xd,yd,ad,bd,eps,cs,PROTOCOL['fixed_iterations'],case['mode'],False)
                save_array('fixed_reference_f',rf);save_array('fixed_reference_g',rg)
                save_array('native_reference_f',nf);save_array('native_reference_g',ng)
                pr=sample_plan(xd,yd,ad,bd,rf,rg,eps,cs,rows,cols)
                pnr=sample_plan(xd,yd,ad,bd,nf,ng,eps,cs,rows,cols)
                plan_check('fixed_author_vs_fp64',pa,pr)
                plan_check('fixed_matched_legacy_vs_fp64',pl,pr)
                plan_check('native_legacy_vs_own_schedule_fp64',pn,pnr)
            # Operator studies use a longer solve for dense/HVP workloads.
            iterations=PROTOCOL['dense_solve_iterations'] if dense else (
                PROTOCOL['large_hvp_solve_iterations'] if case['tier']=='benchmark_hvp' else PROTOCOL['fixed_iterations'])
            if iterations!=PROTOCOL['fixed_iterations']:
                f,g,used=author(iterations)
                old=matched_legacy(x,y,a,b,case,iterations)
                save_array('solve_matched_f',old.f);save_array('solve_matched_g',old.g)
                plan_check('solve_author_vs_matched_legacy',samples(f,g),samples(old.f,old.g))
            report['operator_iterations']=iterations
            report['operator_updates']=used
            save_array('operator_f',f);save_array('operator_g',g)
            p_rows,p_cols=samples(f,g)
            if dense:
                residual=max(float((p_rows.sum(1)-ad).abs().sum()),float((p_cols.sum(0)-bd).abs().sum()))
                rf,rg=fixed_potentials(xd,yd,ad,bd,eps,cs,iterations,case['mode'])
                save_array('solve_reference_f',rf);save_array('solve_reference_g',rg)
                pr=sample_plan(xd,yd,ad,bd,rf,rg,eps,cs,rows,cols)
                plan_check('solve_author_vs_fp64', (p_rows,p_cols),pr)
                rr=max(float((pr[0].sum(1)-ad).abs().sum()),float((pr[1].sum(0)-bd).abs().sum()))
                report['convergence']['kind']='full_CPU_FP64_reconstruction'
                report['convergence']['author_marginal_l1']=residual
                report['convergence']['reference_marginal_l1']=rr
                report['convergence']['confirmed']=residual<=PROTOCOL['marginal_limit'] and rr<=PROTOCOL['reference_marginal_limit']
                if not report['convergence']['confirmed']:
                    report['coverage_gaps'].append('Forward marginal not confirmed at fixed solve budget')
            else:
                report['convergence']={'kind':'sampled_rows_and_columns_only','confirmed':False,
                    'max_sampled_row_relative_mass_error':float(((p_rows.sum(1)-ad[rows])/ad[rows]).abs().max()),
                    'max_sampled_col_relative_mass_error':float(((p_cols.sum(0)-bd[cols])/bd[cols]).abs().max())}
                report['coverage_gaps'].append('Large-shape checks sample rows/columns; no full marginal certificate or full FP64 solve')
            shifted_f=f-cs*x.square().sum(1);shifted_g=g-cs*y.square().sum(1)
            shared=SinkhornResult(x,y,a,b,shifted_f/eps+a.log(),shifted_g/eps+b.log(),
                eps,cs,used,'triton',case['precision'],32,64)
            gpu_marginals={}
            for axis,values,expected,indices in [(1,y-0.5,p_rows@(yd-0.5),rows),(0,x-0.5,p_cols.T@(xd-0.5),cols)]:
                aa=apply_plan_mat_flashstyle(x,y,shifted_f,shifted_g,a.log(),b.log(),values,
                    axis=axis,eps=eps,cost_scale=cs,allow_tf32=case['precision']=='tf32',use_exp2=True,autotune=False)
                ll=apply_plan(shared,values,transpose=axis==0)
                assert torch.isfinite(aa).all() and torch.isfinite(ll).all(), 'Nonfinite matrix apply'
                save_array(f'author_apply_{axis}',aa[indices]);save_array(f'legacy_apply_{axis}',ll[indices])
                compare(f'author_apply_{axis}',aa[indices].cpu(),expected,limit)
                compare(f'legacy_apply_{axis}',ll[indices].cpu(),expected,limit)
                ones=torch.ones_like(b if axis else a)
                am=apply_plan_vec_flashstyle(x,y,shifted_f,shifted_g,a.log(),b.log(),ones,
                    axis=axis,eps=eps,cost_scale=cs,allow_tf32=case['precision']=='tf32',use_exp2=True)
                lm=apply_plan(shared,ones,transpose=axis==0)
                assert torch.isfinite(am).all() and torch.isfinite(lm).all(), 'Nonfinite vector apply'
                mass=p_rows.sum(1) if axis else p_cols.sum(0)
                save_array(f'author_mass_{axis}',am[indices]);save_array(f'legacy_mass_{axis}',lm[indices])
                compare(f'author_mass_{axis}',am[indices].cpu(),mass,limit)
                compare(f'legacy_mass_{axis}',lm[indices].cpu(),mass,limit)
                gpu_marginals[str(axis)]=float((am-(a if axis else b)).abs().sum())
            report['diagnostics']['full_GPU_FP32_marginal_l1']=max(gpu_marginals.values())
            ga,gb=sinkhorn_geomloss_online_grad_sqeuclid(x,y,a,b,f,g,eps=eps,cost_scale=cs,
                allow_tf32=case['precision']=='tf32',use_exp2=True,autotune=False)
            gl,gm=source_gradient(shared),target_gradient(shared)
            assert all(torch.isfinite(t).all() for t in (ga,gb,gl,gm)), 'Nonfinite gradient'
            for side,author_grad,legacy_grad,p,points,keys,weights,indices in [
                ('x',ga,gl,p_rows,xd,yd,ad,rows),('y',gb,gm,p_cols.T,yd,xd,bd,cols)]:
                mass=p.sum(1);mean=(p@keys)/mass[:,None]
                normalized=2*cs*weights[indices,None]*(points[indices]-mean)
                actual_mass=2*cs*(mass[:,None]*points[indices]-p@keys)
                save_array('author_gradient_'+side,author_grad[indices]);save_array('legacy_gradient_'+side,legacy_grad[indices])
                compare('author_normalized_gradient_'+side,author_grad[indices].cpu(),normalized,limit)
                compare('legacy_actual_mass_gradient_'+side,legacy_grad[indices].cpu(),actual_mass,limit)
                report['diagnostics']['gradient_convention_difference_'+side]=relative(normalized,actual_mass)
            if dense or case['tier']=='benchmark_hvp':
                fo,go=f+eps*a.log(),g+eps*b.log()
                save_array('hvp_ott_f',fo);save_array('hvp_ott_g',go)
                shared_hvp=SinkhornResult(x,y,a,b,(fo-cs*x.square().sum(1))/eps,
                    (go-cs*y.square().sum(1))/eps,eps,cs,used,'triton','ieee',32,64)
                cap=PROTOCOL['dense_cg_cap'] if dense else PROTOCOL['large_cg_cap']
                ha,ia=hvp_x_sqeuclid_from_potentials(x,y,fo,go,v,eps=eps,cost_scale=cs,
                    tau2=PROTOCOL['tau2'],max_cg_iter=cap,cg_rtol=PROTOCOL['cg_rtol'],cg_atol=PROTOCOL['cg_atol'],
                    use_preconditioner=False,allow_tf32=False,use_exp2=True,autotune=False)
                hl,il=hessian_vector_product(shared_hvp,v,damping=eps*PROTOCOL['tau2'],
                    max_cg_iters=cap,cg_rtol=PROTOCOL['cg_rtol'],cg_atol=PROTOCOL['cg_atol'],return_info=True)
                assert torch.isfinite(ha).all() and torch.isfinite(hl).all(), 'Nonfinite HVP'
                report['hvp']={'author_cg':asdict(ia),'legacy_cg':asdict(il),'cap':cap,
                    'tau2':PROTOCOL['tau2'],'legacy_schur_damping':eps*PROTOCOL['tau2'],
                    'interpretation':'Damped implicit operator at supplied plan; optimal Hessian requires converged forward',
                    'sampled_relative_l2':relative(ha[rows].cpu().double(),hl[rows].cpu().double())}
                save_array('author_hvp',ha[rows]);save_array('legacy_hvp',hl[rows])
                confirmed=cg_confirmed(asdict(ia)) and cg_confirmed(asdict(il))
                if confirmed:
                    compare('hvp_author_vs_legacy',ha[rows].cpu(),hl[rows].cpu(),PROTOCOL['hvp_relative_limit'])
                else:
                    report['coverage_gaps'].append('HVP CG not confirmed in both backends; parity metric is diagnostic only')
                if dense:
                    hp=torch.exp((fo.cpu().double()[:,None]+go.cpu().double()[None,:]-cost_matrix(xd,yd,cs))/eps)
                    hd,linear=direct_hvp(xd,yd,hp,vd,eps,cs,PROTOCOL['tau2'])
                    save_array('direct_hvp',hd)
                    report['hvp']['direct_relative_residual']=linear
                    assert linear<=PROTOCOL['linear_residual_limit'], 'FP64 linear solve residual too large'
                    if confirmed:
                        compare('hvp_author_vs_direct',ha.cpu(),hd,PROTOCOL['hvp_relative_limit'])
                        compare('hvp_legacy_vs_direct',hl.cpu(),hd,PROTOCOL['hvp_relative_limit'])
            report['warnings']=[{'category':w.category.__name__,'message':str(w.message)} for w in caught]
    except Exception as exc:
        if 'caught' in locals():
            report['warnings']=[{'category':w.category.__name__,'message':str(w.message)} for w in caught]
        report['error']=f'{type(exc).__name__}: {exc}'
        report['traceback']=traceback.format_exc()
        print(report['traceback'],flush=True)
    finally:
        report['status']=decide(report['checks'],report['coverage_gaps'],report.get('error'))
        def safe(v):
            if isinstance(v,float) and not math.isfinite(v):return None
            if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
            if isinstance(v,list):return [safe(x) for x in v]
            return v
        np.savez_compressed(args.output/'arrays.npz',**arrays)
        (args.output/'result.json').write_text(json.dumps(safe(report),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f"[step4-case] {case['id']} {case['shape']} {case['mode']} {case['precision']}: {report['status']}",flush=True)
    return 1 if report['status']=='failed' else 0


if __name__=='__main__':
    with torch.no_grad():
        raise SystemExit(main())
