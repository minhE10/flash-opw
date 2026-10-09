"""Run the declared step-4 inventory in isolated GPU processes and bundle evidence."""
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

try:
    from .author_flashsinkhorn_sources import ROOT,verify
    from .author_flashsinkhorn_profiles import prepare_profile,verify_profile
    from .author_step4_reference import PROTOCOL,inventory
except ImportError:
    from author_flashsinkhorn_sources import ROOT,verify
    from author_flashsinkhorn_profiles import prepare_profile,verify_profile
    from author_step4_reference import PROTOCOL,inventory


def legacy_hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'flashsinkhorn').rglob('*.py'))}


def main():
    device=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
    if not device or ',' in device or device=='-1':
        raise SystemExit('Set CUDA_VISIBLE_DEVICES to exactly one allocated GPU')
    output=ROOT/'outputs'/('author_flashsinkhorn_step4_'+datetime.now().strftime('%Y%m%d_%H%M%S')+f'_{os.getpid()}')
    output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(ROOT/'scripts'),FLASHOPW_AUTOTUNE='0',FLASHOPW_GRADIENT_KERNEL='1')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        env[key]='2'
    cases=inventory()
    summary={'status':'running','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Step 4 correctness only: dense FP64 studies and sampled benchmark-shape operators',
        'protocol':PROTOCOL,'inventory':cases,'source':verify(),'legacy_before':legacy_hashes(),
        'runs':[],'limitations':['Large cases use sampled FP64 rows/columns, not a full dense reference.',
            'Matched legacy symmetric boundaries are a validation adapter; native legacy is recorded separately.',
            'HVP damping is matched as legacy damping = epsilon * author tau2.',
            'Gradient conventions are checked separately at nonconverged plans.',
            'No changes to original author or legacy source; not a timing benchmark.',
            'Step 3 external OTT-Hessian coverage gaps remain unresolved.']}
    helper_names=['author_step4_reference.py','check_author_step4.py','run_author_step4.py','author_flashsinkhorn_fp64.py']
    summary['validation_code_sha256']={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in helper_names}
    def save():
        (output/'step4-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    def run(command,log_path):
        with log_path.open('w',encoding='utf-8') as log:
            child=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            done=threading.Event();start=time.monotonic()
            def beat():
                while not done.wait(30):
                    if child.poll() is None:
                        print(f'[step4] PID {child.pid} running ({time.monotonic()-start:.0f}s); {log_path}',flush=True)
            thread=threading.Thread(target=beat,daemon=True);thread.start()
            try:
                for line in child.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                return child.wait()
            except KeyboardInterrupt:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
            finally:
                done.set();thread.join(timeout=1)
    print(f'[step4] Run: {output}; cases: {len(cases)}',flush=True)
    save()
    implementation=None
    try:
        if summary['source']['status']!='verified':raise ValueError('Author source integrity failed')
        summary['environment']={name:metadata.version(name) for name in ('torch','triton','numpy','geomloss')}
        if summary['environment']['geomloss']!='0.3.1':raise ValueError('Use the unified step 3 environment with GeomLoss 0.3.1')
        for command,name in [([sys.executable,'-m','pip','freeze'],'pip-freeze.txt'),
                             (['git','rev-parse','HEAD'],'project-commit.txt'),
                             (['git','status','--short'],'project-status.txt'),
                             (['nvidia-smi','-i',device],'nvidia-smi.txt')]:
            record=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True,timeout=60)
            (output/name).write_text(record.stdout+record.stderr,encoding='utf-8')
        profile_dir=output/'profile';profile_dir.mkdir()
        implementation,summary['profile']=prepare_profile(profile_dir)
        save()
        for i,case in enumerate(cases,1):
            print(f"[step4] {i}/{len(cases)}: {case['id']} {case['tier']} {case['shape']} {case['mode']} {case['precision']}",flush=True)
            destination=output/case['id']
            code=run([sys.executable,'-u','-B',ROOT/'scripts/check_author_step4.py',
                '--implementation',implementation,'--case-id',case['id'],'--output',destination],output/(case['id']+'.log'))
            path=destination/'result.json'
            record=json.loads(path.read_text()) if path.exists() else {'status':'missing_report'}
            summary['runs'].append({'id':case['id'],'exit_code':code,'status':record['status']})
            save()
            if verify_profile(implementation)['status']!='profile_verified':
                raise ValueError('Compatibility profile changed during run')
            if code and (record.get('status')=='missing_report' or 'out of memory' in record.get('error','').lower()):
                raise RuntimeError('Incomplete child or OOM: stop; unrun cases are not validated')
        failed=[r['id'] for r in summary['runs'] if r['exit_code'] or r['status']=='failed']
        summary['failed_cases']=failed
        summary['coverage_gaps']=[r['id'] for r in summary['runs'] if r['status']=='passed_checks_with_coverage_gaps']
        summary['status']='failed' if failed else 'passed_checks_with_coverage_gaps' if summary['coverage_gaps'] else 'passed_checks'
    except KeyboardInterrupt:
        summary.update(status='interrupted',error='Interrupted; unfinished cases are not validated')
    except Exception as exc:
        summary.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        summary['source_after']=verify();summary['legacy_after']=legacy_hashes()
        summary['profile_after']=verify_profile(implementation) if implementation else {'status':'not_prepared'}
        if (summary['source_after']['status']!='verified' or summary['legacy_before']!=summary['legacy_after']
            or summary['profile_after']['status']!='profile_verified'):
            summary['status']='failed'
        summary['missing_cases']=sorted({c['id'] for c in cases}-{r['id'] for r in summary['runs']})
        save()
        paths=[p for p in output.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.log','.txt','.patch')
               and not any(part.startswith('implementation-') for part in p.relative_to(output).parts)]
        hashes={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        checksum=output/'artifact-sha256.json';checksum.write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
        with zipfile.ZipFile(output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for p in paths+[checksum]:z.write(p,p.relative_to(output).as_posix())
        print(f'[step4] Artifact bundle: {output.with_suffix(".zip")}',flush=True)
        print(f"[step4] {summary['status']}; completed {len(summary['runs'])}/{len(cases)}; missing: {len(summary['missing_cases'])}",flush=True)
    return 1 if summary['status'] in ('failed','interrupted') else 3 if summary['status']=='passed_checks_with_coverage_gaps' else 0


if __name__=='__main__':raise SystemExit(main())
