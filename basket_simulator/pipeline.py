"""Fit product preferences first, then count/category corrections.

The structural stage includes its preference parent as an exact starting point.
Validation selection can retain that parent if added structure does not help.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import torch
from .trainable import TrainableBasketModel
from .train import fit,load_data,write_json


def memory_estimate(config):
    """Conservative dense-preference lower bound, not peak RSS certification."""
    preference=(int(config['users'])+int(config['stores']))*int(config['C'])*int(config['P'])*8
    return {'dense_preference_bytes':preference,'lbfgs_history_bytes_10_corrections':20*preference,
            'parameter_gradient_work_and_history_lower_bound_bytes':24*preference,
            'scope':'Excludes covariates, ESP/autograd workspace and other coefficients'}


def response_start(parent):
    """Freeze preferences and expose count/price blocks with a lossless migration.

    The tested response stage uses a dense cross-category coordinate system for
    small catalogues. Large-catalogue price fitting needs its own validation.
    """
    if parent.C>10:raise ValueError('Full response-stage fitting is validated only for C<=10; use explicit low-rank blocks for larger catalogues')
    blocks=['rho','kappa','choice_price','count_price_scale','category_price','cross_p','count_offset']
    blocks+=['linear_p','linear_q'] if hasattr(parent,'linear_p') else ['V']
    if not hasattr(parent,'independent_category_anchor'):blocks.append('inclusive_price_weight')
    cfg={**parent.config,'cross_rank':parent.C,'learn_count_offset':True,'learn_inclusive_price':True,
         'independent_only':False,'trainable_parameters':blocks}
    model=TrainableBasketModel(cfg)
    state={k:v for k,v in parent.state_dict().items() if k not in ('cross_p','cross_q')}
    model.load_state_dict(state,strict=False)
    with torch.no_grad():
        model.cross_p.copy_(parent.cross_p@parent.cross_q.T)
        model.cross_q.copy_(torch.eye(parent.C,dtype=torch.float64))
        if not hasattr(parent,'inclusive_price_weight'):model.inclusive_price_weight.zero_()
    return model


def fit_response(parent,train,validation,out,maxiter=200,maxcalls=200,time_limit=300.,batch_size=512):
    model=response_start(parent)
    with torch.no_grad():
        ctx,y=validation;c={k:v[:128] for k,v in ctx.items()}
        np.testing.assert_allclose(model.log_probability(c,y[:128]),parent.log_probability(c,y[:128]),atol=1e-10)
    result=fit(model,train,validation,out,maxiter=maxiter,maxcalls=maxcalls,time_limit=time_limit,
               batch_size=batch_size,ridge=10.)
    return model,result


def category_scale_start(parent):
    """Add the existing category inclusive-value scale to a frozen response block."""
    cfg={**parent.config,'trainable_parameters':list(dict.fromkeys(
        [*parent.config['trainable_parameters'],'tau']))}
    model=TrainableBasketModel(cfg)
    model.load_state_dict(parent.state_dict())
    return model


def fit_category_scale(parent,train,validation,out,maxiter=200,maxcalls=200,time_limit=300.,batch_size=512):
    model=category_scale_start(parent)
    with torch.no_grad():
        ctx,y=validation;c={k:v[:128] for k,v in ctx.items()}
        np.testing.assert_allclose(model.log_probability(c,y[:128]),parent.log_probability(c,y[:128]),atol=1e-10)
    result=fit(model,train,validation,out,maxiter=maxiter,maxcalls=maxcalls,time_limit=time_limit,
               batch_size=batch_size,ridge=10.)
    return model,result


def independent_start(config,member,seed=0):
    if config['R']!=config['P']:raise ValueError('Exact independent nesting requires R=P')
    cfg={**config,'preference_mode':'direct','category_anchor':config.get('category_anchor','legacy'),
         'learn_inclusive_price':True,'independent_only':True,'seed':seed}
    cfg.pop('trainable_parameters',None)
    m=TrainableBasketModel(cfg,seed)
    incidence=member.detach().numpy().mean(0).ravel()
    initial=np.log(np.clip(incidence,1e-6,1-1e-6)/np.clip(1-incidence,1e-6,1))
    def population(v):
        A=np.logaddexp(0.,v).sum();Z=-np.expm1(-A)
        score=A+np.log(Z)-incidence@v
        return float(score),1/(1+np.exp(-v))/Z-incidence
    pooled=minimize(population,np.clip(initial,-15,15),jac=True,method='L-BFGS-B',
                    bounds=[(-15.,15.)]*len(initial),options={'gtol':1e-8,'ftol':1e-12})
    with torch.no_grad():
        m.lam.copy_(torch.tensor(pooled.x));m.mu.zero_();m.choice_price.zero_()
        m.rho.zero_();m.count_price_scale.fill_(1.);m.category_price.zero_()
        m.kappa.zero_();m.tau.fill_(np.log(np.expm1(1.)));m.cross_p.zero_();m.omega.zero_()
        m.V.zero_()
        if hasattr(m,'linear_p'):m.linear_p.zero_()
    return m


def staged_fit(config,train,validation,out,ridges=(1.,10.),seed=0,maxiter=200,
               maxcalls=200,time_limit=300.,batch_size=512):
    if config.get('fit_category_scale',False) and not config.get('fit_response',False):
        raise ValueError('fit_category_scale requires fit_response')
    estimate=memory_estimate(config)
    budget=config.get('memory_budget_mib')
    if budget is not None:
        if not np.isfinite(budget) or budget<=0:raise ValueError('Positive memory_budget_mib required')
        if estimate['parameter_gradient_work_and_history_lower_bound_bytes']>budget*2**20:
            raise ValueError('Dense-preference optimizer lower bound exceeds memory_budget_mib; reduce cohort/catalogue or use a validated factor/sparse fitting route')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'memory_estimate.json',estimate)
    if not ridges or any(r<0 for r in ridges):raise ValueError('Nonnegative ridge choices required')
    records=[];selected=None;best=-np.inf
    for i,ridge in enumerate(ridges):
        model=independent_start(config,train[1],seed)
        row=fit(model,train,validation,out/f'preferences_{i}',maxiter=maxiter,
                ridge=ridge,batch_size=batch_size,maxcalls=maxcalls,time_limit=time_limit)
        row.update(ridge=float(ridge),run=f'preferences_{i}')
        records.append(row)
        if row['best_validation_ll']>best:
            best=row['best_validation_ll'];selected=model
    parameters=['rho','kappa','linear_p','linear_q'] if hasattr(selected,'linear_p') else ['rho','kappa','V']
    if hasattr(selected,'count_offset'):parameters.append('count_offset')
    cfg={**selected.config,'independent_only':False,'trainable_parameters':parameters}
    structure=TrainableBasketModel(cfg,seed)
    structure.load_state_dict(selected.state_dict())
    correction=fit(structure,train,validation,out/'structure',maxiter=maxiter,ridge=10.,
                   batch_size=batch_size,maxcalls=maxcalls,time_limit=time_limit)
    response=None;category_calibration=None;selected_path=out/'structure'
    if config.get('fit_response',False):
        structure,response=fit_response(structure,train,validation,out/'response',maxiter=maxiter,
            maxcalls=maxcalls,time_limit=time_limit,batch_size=batch_size)
        selected_path=out/'response'
    if config.get('fit_category_scale',False):
        if not response['final_stationarity_passed']:
            write_json(out/'stopped.json',{'stage':'response','decision':'Category calibration stopped: inspect response fit/history before another fit.'})
            raise RuntimeError('Response final stationarity failed; category calibration was not launched')
        structure,category_calibration=fit_category_scale(structure,train,validation,out/'category_calibration',
            maxiter=maxiter,maxcalls=maxcalls,time_limit=time_limit,batch_size=batch_size)
        selected_path=out/'category_calibration'
    for name in ('best.pt','config.json'):shutil.copy2(selected_path/name,out/name)
    result={'preferences':records,'selected_preference_validation_ll':best,'structure':correction,
            'structure_validation_ll_gain':correction['best_validation_ll']-best,
            'structural_parent_retained':correction['best_iteration']==0,
            'stationarity_scope':'Stage-specific active parameters. No joint optimum or causal certificate.',
            'model':'Direct household/product preferences then ESP count/category-pair corrections',
            'response':response,'category_calibration':category_calibration,'selected_stage':selected_path.name}
    write_json(out/'selection.json',result)
    return structure,result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--ridges',default='1,10');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--maxiter',type=int,default=200);p.add_argument('--maxcalls',type=int,default=200)
    p.add_argument('--time-limit',type=float,default=300.);p.add_argument('--batch-size',type=int,default=512)
    args=p.parse_args();torch.set_num_threads(1)
    manifest={'files':{
        str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (args.train,args.validation,args.config)},
        'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in Path(__file__).parent.glob('*.py')}}
    config=json.loads(Path(args.config).read_text())
    train=load_data(args.train,config);validation=load_data(args.validation,config)
    model,result=staged_fit(config,train,validation,args.out,
        ridges=tuple(float(v) for v in args.ridges.split(',')),seed=args.seed,maxiter=args.maxiter,
        maxcalls=args.maxcalls,time_limit=args.time_limit,batch_size=args.batch_size)
    write_json(Path(args.out)/'input_manifest.json',manifest)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
