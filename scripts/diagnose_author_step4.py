"""Summarize step-4 evidence and test a CPU TF32 truncation hypothesis.

This is a diagnostic model, not a CUDA rerun or an alternate pass criterion.
Never executes code from the archive. Use audit_author_step4.py separately.
"""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import zipfile
import numpy as np
import torch

try:
    from .author_step4_reference import inputs,tensor_hashes,relative,cost_matrix
except ImportError:
    from author_step4_reference import inputs,tensor_hashes,relative,cost_matrix


def trunc_tf32(t):
    """Model truncating the bottom 13 FP32 fraction bits, preserving sign."""
    bits=t.contiguous().numpy().view(np.uint32)
    return torch.from_numpy((bits & np.uint32(0xffffe000)).view(np.float32).copy()).double()


def diagnose(archive):
    torch.set_num_threads(2)
    with zipfile.ZipFile(archive) as z:
        summary=json.loads(z.read('step4-summary.json'))
        records=[json.loads(z.read(r['id']+'/result.json')) for r in summary['runs']]
        result={'reported_status':summary['status'],'counts':dict(Counter(r['status'] for r in records)),
            'tiers':{},'failure_categories':dict(Counter(k for r in records for k,v in r['checks'].items() if not v['passed'])),
            'runtime_errors':[dict(id=r['case']['id'],error=r['error']) for r in records if r.get('error')],
            'cg_unconfirmed':[],'convergence_confirmed':[],'input_hash_mismatches':{},
            'case_metrics':[],'tf32_models':[],
            'limitation':'TF32 truncation is a CPU diagnostic hypothesis; does not reproduce all Triton reductions or certify CUDA correctness.'}
        for tier in ('dense','benchmark_forward','benchmark_hvp'):
            group=[r for r in records if r['case']['tier']==tier]
            result['tiers'][tier]=dict(Counter(r['status'] for r in group))
        for record in records:
            case=record['case'];raw=inputs(case)
            hashes=tensor_hashes(raw)
            result['input_hash_mismatches'][case['id']]=[k for k,h in hashes.items() if h!=record['input_sha256'][k]]
            if record.get('hvp'):
                for backend in ('author','legacy'):
                    if not record['hvp'][backend+'_cg']['cg_converged']:
                        result['cg_unconfirmed'].append([case['id'],backend])
            if record['convergence'].get('confirmed'):result['convergence_confirmed'].append(case['id'])
            with np.load(BytesIO(z.read(case['id']+'/arrays.npz')),allow_pickle=False) as saved:
                arrays={k:torch.from_numpy(saved[k].copy()) for k in saved.files}
            metrics={'id':case['id'],'shape':case['shape'],'mode':case['mode'],'precision':case['precision'],
                'tier':case['tier'],'status':record['status'],
                'failed_checks':{k:v['relative_l2'] for k,v in record['checks'].items() if not v['passed']},
                'author_vs_matched_plan':record['checks']['fixed_author_vs_matched_legacy_rows']['relative_l2'],
                'author_vs_native_plan':record['diagnostics']['native_schedule_plan_relative_l2']}
            for kind in ('apply_1','apply_0','mass_1','mass_0','gradient_x','gradient_y','hvp'):
                if 'author_'+kind in arrays and 'legacy_'+kind in arrays:
                    metrics['cross_'+kind]=relative(arrays['author_'+kind].double(),arrays['legacy_'+kind].double())
            if 'author_hvp' in arrays:
                metrics['author_hvp_norm']=float(arrays['author_hvp'].double().norm())
                metrics['legacy_hvp_norm']=float(arrays['legacy_hvp'].double().norm())
                if 'direct_hvp' in arrays:
                    metrics['stored_direct_hvp_norm']=float(arrays['direct_hvp'].double().norm())
            result['case_metrics'].append(metrics)
            # Representative scales: sensitive small-eps case and all benchmark
            # dimensions at n=20000, plus the largest high-dimensional case.
            representative=(case['id'] in ('case_001','case_013','case_093') or
                (case['tier']=='benchmark_forward' and case['shape'][0]==20000 and case['mode']=='alternating'))
            if not representative:continue
            assert all(hashes[k]==record['input_sha256'][k] for k in ('x','y','a','b'))
            x,y,a,b=[t.double() for t in raw[:4]]
            qx,qy=trunc_tf32(raw[0]),trunc_tf32(raw[1])
            rows=arrays['rows'].long();f,g=arrays['operator_f'].double(),arrays['operator_g'].double()
            eps,cs=case['eps'],case['cost_scale']
            exact_cost=cost_matrix(x[rows],y,cs)
            # Norm terms remain IEEE FP32 in the actual kernels. Here FP64
            # norms isolate the large coordinate-dot precision effect.
            model_cost=cs*(x[rows].square().sum(1)[:,None]+y.square().sum(1)[None,:]-2*(qx[rows]@qy.T))
            exponent=a[rows].log()[:,None]+b.log()[None,:]+(f[rows,None]+g[None,:])/eps
            pe=torch.exp(exponent-exact_cost/eps);pq=torch.exp(exponent-model_cost/eps)
            observed=arrays['author_mass_1'].double()
            result['tf32_models'].append({'id':case['id'],'shape':case['shape'],'eps':eps,
                'recorded_mass_error':record['checks']['author_mass_1']['relative_l2'],
                'exact_mass_error_recomputed':relative(observed,pe.sum(1)),
                'trunc_model_mass_error':relative(observed,pq.sum(1)),
                'median_dot_cost_bias_over_eps':float(((model_cost-exact_cost)/eps).median()),
                'trunc_model_plan_vs_exact':relative(pq,pe)})
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=diagnose(args.archive)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for key in ('counts','tiers','runtime_errors','cg_unconfirmed','tf32_models'):
        print(key,json.dumps(result[key]))


if __name__=='__main__':main()
