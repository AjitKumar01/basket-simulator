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
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
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
    cfg={**selected.config,'independent_only':False,'trainable_parameters':parameters}
    structure=TrainableBasketModel(cfg,seed)
    structure.load_state_dict(selected.state_dict())
    correction=fit(structure,train,validation,out/'structure',maxiter=maxiter,ridge=10.,
                   batch_size=batch_size,maxcalls=maxcalls,time_limit=time_limit)
    for name in ('best.pt','config.json'):shutil.copy2(out/'structure'/name,out/name)
    result={'preferences':records,'selected_preference_validation_ll':best,'structure':correction,
            'structure_validation_ll_gain':correction['best_validation_ll']-best,
            'structural_parent_retained':correction['best_iteration']==0,
            'stationarity_scope':'Stage-specific active parameters. No joint optimum or causal certificate.',
            'model':'Direct household/product preferences then ESP count/category-pair corrections'}
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
