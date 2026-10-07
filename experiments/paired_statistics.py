"""Paired query uncertainty conditional on a fixed gallery and frozen models."""

import numpy as np
from scipy.stats import binomtest, bootstrap


def paired_query_statistics(left, right, labels, *, resamples=10000, seed=20261007):
    labels = np.asarray(labels)
    a,b = [np.asarray(e["average_precision"],dtype=float) for e in (left,right)]
    if (a.shape != labels.shape or b.shape != labels.shape or labels.ndim != 1
            or len(labels) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all()
            or min(resamples,seed) < 0 or resamples < 100):
        raise ValueError("Finite paired AP arrays, >=2 queries and >=100 resamples required")
    delta = a-b
    # Resampling paired differences is equivalent to jointly resampling AP pairs.
    interval = bootstrap((delta,),np.mean,method="percentile",confidence_level=.95,
                         n_resamples=resamples,batch=256,rng=np.random.default_rng(seed)).confidence_interval
    predictions = [np.asarray(e["predictions"]["1"]) for e in (left,right)]
    if any(p.shape != labels.shape for p in predictions):
        raise ValueError("Paired prediction lengths do not match query labels")
    ac,bc = [p == labels for p in predictions]
    left_only,right_only = int(np.sum(ac&~bc)),int(np.sum(~ac&bc))
    discordant = left_only+right_only
    p = float(binomtest(left_only,discordant,p=.5,alternative="two-sided").pvalue) if discordant else 1.
    return dict(queries=len(labels),delta_MAP_pp=100*float(delta.mean()),
        map_difference_ci95_pp=[100*float(interval.low),100*float(interval.high)],
        delta_ACC1_pp=100*float(ac.mean()-bc.mean()),both_correct=int(np.sum(ac&bc)),
        left_only_correct=left_only,right_only_correct=right_only,both_wrong=int(np.sum(~ac&~bc)),
        mcnemar_exact_two_sided_p=p,bootstrap_resamples=resamples,bootstrap_seed=seed,
        bootstrap_method="paired query percentile; fixed gallery/parameters; unadjusted 95% CI")


def holm_adjust(pvalues):
    """Holm step-down adjustment, restoring the original hypothesis order."""
    values = np.asarray(pvalues,dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError("P-values must be a finite vector in [0,1]")
    order = np.argsort(values,kind="stable")
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1,np.maximum.accumulate(values[order]*np.arange(len(values),0,-1)))
    return adjusted.tolist()
