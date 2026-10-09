"""Step 4 protocol and CPU FP64 equations; no author/legacy kernel imports."""
import hashlib
import math
import torch

PROTOCOL = {
    "version": 1, "fixed_iterations": 10, "dense_solve_iterations": 500,
    "large_hvp_solve_iterations": 100, "sample_count": 8,
    "tau2": 1e-5, "cg_rtol": 1e-6, "cg_atol": 1e-6,
    "dense_cg_cap": 256, "large_cg_cap": 50,
    "ieee_relative_limit": 5e-4, "tf32_relative_limit": 5e-3,
    "hvp_relative_limit": 5e-4, "linear_residual_limit": 1e-9,
    "marginal_limit": 1e-3,
    "reference_marginal_limit": 1e-6,
    "matrix_values": "coordinates_minus_half",
}


def inventory():
    cases = []
    shapes = [(17,23,d) for d in (1,4,8,16,32,64,128,256,512,1024)]
    shapes += [(131,127,1),(128,128,64),(257,193,32),(512,640,13)]
    for i, shape in enumerate(shapes):
        for mode in ("symmetric", "alternating"):
            for precision in ("ieee", "tf32"):
                cases.append(dict(tier="dense",shape=list(shape),mode=mode,precision=precision,
                    eps=(0.01,0.1,1.0)[i%3],cost_scale=(1.0,0.5)[i%2],
                    weights="nonuniform" if i%2 else "uniform",seed=20261009+i,
                    distribution="uniform_dimension_scaled"))
    sizes = (5000,10000,20000,30000,40000,50000)
    dims = (4,8,16,32,64,128,256,512,1024)
    shapes = sorted({(n,n,64) for n in sizes} | {(20000,20000,d) for d in dims}
                    | {(n,n,1024) for n in sizes})
    for shape in shapes:
        for mode in ("symmetric", "alternating"):
            cases.append(dict(tier="benchmark_forward",shape=list(shape),mode=mode,precision="tf32",
                eps=0.1,cost_scale=1.0,weights="uniform",seed=0,distribution="uniform_unit_cube"))
    hvp_shapes = sorted({(n,n,64) for n in (5000,6000,7000,8000,9000,10000,20000,30000,40000,50000)}
                        | {(10000,10000,d) for d in (4,8,16,32,64,128,256,512)})
    for shape in hvp_shapes:
        cases.append(dict(tier="benchmark_hvp",shape=list(shape),mode="symmetric",precision="ieee",
            eps=0.1,cost_scale=1.0,weights="uniform",seed=0,distribution="uniform_unit_cube"))
    for i, case in enumerate(cases):
        case["id"] = f"case_{i:03d}"
    return cases


def inputs(case):
    n,m,d = case["shape"]
    gen = torch.Generator(device="cpu").manual_seed(case["seed"])
    scale = 1/math.sqrt(d) if case["distribution"] == "uniform_dimension_scaled" else 1.0
    x,y = torch.rand(n,d,generator=gen,dtype=torch.float32)*scale, torch.rand(m,d,generator=gen,dtype=torch.float32)*scale
    if case["weights"] == "uniform":
        a,b = torch.full((n,),1/n,dtype=torch.float32),torch.full((m,),1/m,dtype=torch.float32)
    else:
        a,b = torch.rand(n,generator=gen,dtype=torch.float32)+0.2,torch.rand(m,generator=gen,dtype=torch.float32)+0.2
        a,b = a/a.sum(), b/b.sum()
    v = torch.randn(n,d,generator=gen,dtype=torch.float32)
    return x,y,a,b,v


def tensor_hashes(tensors):
    return {name:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name,t in zip(("x","y","a","b","direction"),tensors)}


def selected(n, dense):
    return torch.arange(n) if dense else torch.linspace(0,n-1,PROTOCOL["sample_count"]).long().unique()


def cost_matrix(x,y,scale):
    # Avoid n*m*d storage: FP64 Gram evaluation, with no clipping of costs.
    return scale*(x.square().sum(1)[:,None]+y.square().sum(1)[None,:]-2*(x@y.T))


def fixed_potentials(x,y,a,b,eps,scale,iterations,mode,author_schedule=True):
    c = cost_matrix(x,y,scale)
    la,lb = a.log(),b.log()
    f,g = torch.zeros_like(a),torch.zeros_like(b)
    def step(f,g,alpha):
        ff = -eps*torch.logsumexp(lb[None,:]+(g[None,:]-c)/eps,1)
        gg = -eps*torch.logsumexp(la[:,None]+((ff if mode=="alternating" else f)[:,None]-c)/eps,0)
        return (1-alpha)*f+alpha*ff,(1-alpha)*g+alpha*gg
    if mode=="symmetric" and author_schedule:
        f,g=step(f,g,1.0)
    for _ in range(iterations):
        f,g=step(f,g,0.5 if mode=="symmetric" else 1.0)
    if mode=="symmetric" and author_schedule:
        f,g=step(f,g,1.0)
    return f,g


def sample_plan(x,y,a,b,f,g,eps,scale,rows,cols):
    row = torch.exp(a[rows].log()[:,None]+b.log()[None,:]
                    +(f[rows,None]+g[None,:]-cost_matrix(x[rows],y,scale))/eps)
    col = torch.exp(a.log()[:,None]+b[cols].log()[None,:]
                    +(f[:,None]+g[None,cols]-cost_matrix(x,y[cols],scale))/eps)
    return row,col


def relative(actual, expected):
    return float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected).clamp_min(1e-30))


def check(actual, expected, limit):
    value=relative(actual,expected)
    return {"relative_l2":value,"limit":limit,"passed":math.isfinite(value) and value<=limit}


def cg_confirmed(info):
    r,initial=info.get("cg_residual"),info.get("cg_initial_residual")
    return (info.get("cg_converged") is True and
            all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in (r,initial)) and
            r<=max(PROTOCOL["cg_atol"],PROTOCOL["cg_rtol"]*initial))


def decide(checks, gaps, error=None):
    if error or not checks or any(not c["passed"] for c in checks.values()):
        return "failed"
    return "passed_checks_with_coverage_gaps" if gaps else "passed_checks"
