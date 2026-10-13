"""Train from context/member NPZ arrays using bounded full-batch L-BFGS-B."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import torch
from .common import TrainableBasketModel

FIELDS=('house','store','week','dlp','disp','mail')


def load_data(path,config):
    with np.load(path,allow_pickle=False) as archive:
        missing=set(FIELDS+('member',))-set(archive.files)
        if missing:raise ValueError(f'Missing arrays: {sorted(missing)}')
        arrays={k:archive[k] for k in FIELDS+('member',)}
    C,P,R=(int(config[k]) for k in ('C','P','R'));N=len(arrays['house'])
    if N<1:raise ValueError('Dataset must be nonempty')
    for k,a in arrays.items():
        shape=(N,) if k in ('house','store','week') else (N,C,P)
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'{k}: require finite shape {shape}')
    for k,size in (('house',config['users']),('store',config['stores'])):
        a=arrays[k]
        if not np.equal(a,a.astype(np.int64)).all() or a.min()<0 or a.max()>=size:
            raise ValueError(f'{k}: IDs must be integers in [0,{size})')
    y=arrays['member']
    if not np.isin(y,(0,1)).all() or (y.sum((1,2))==0).any() or (y.sum(-1)>R).any():
        raise ValueError('member must be binary, nonempty, and within per-category support')
    ctx={k:torch.tensor(arrays[k],dtype=torch.long if k in ('house','store') else torch.float64) for k in FIELDS}
    return ctx,torch.tensor(y,dtype=torch.float64)


def subset(ctx,idx):return {k:v[idx] for k,v in ctx.items()}


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def fit(model,train,validation,out,maxiter=200,ridge=10.,batch_size=512,threshold=2e-4,
        maxcalls=None,time_limit=None,scales=None):
    """Write best/final states and diagnostics. Existing run directories are rejected."""
    if maxiter<1 or ridge<0 or batch_size<1 or threshold<=0:raise ValueError('Invalid fit settings')
    if (maxcalls is not None and maxcalls<1) or (time_limit is not None and time_limit<=0):
        raise ValueError('Fit budgets must be positive')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    history_size=int(model.config.get('lbfgs_history',10))
    if not 1<=history_size<=20:raise ValueError('lbfgs_history must be between1 and20')
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad];raw_bounds=model.parameter_bounds()
    scales=np.ones(sum(p.numel() for _,p in parameters)) if scales is None else np.asarray(scales)
    if scales.shape!=(sum(p.numel() for _,p in parameters),) or not np.isfinite(scales).all() or not (scales>0).all():
        raise ValueError('Require finite positive coordinate scale for every active parameter')
    bounds=[(None if lo is None else lo/s,None if hi is None else hi/s) for (lo,hi),s in zip(raw_bounds,scales)]
    from basket_simulator.train import gradient_norm as raw_norm
    def gradient_norm(v,g,b):return raw_norm(v*scales,g/scales,raw_bounds)
    last_evaluation={}
    if not parameters:raise ValueError('No trainable parameters')
    def vector():return np.concatenate([p.detach().numpy().ravel() for _,p in parameters])/scales
    def assign(v):
        v=v*scales
        offset=0
        with torch.no_grad():
            for _,p in parameters:
                p.copy_(torch.from_numpy(v[offset:offset+p.numel()].reshape(p.shape)));offset+=p.numel()
    def objective(v):
        assign(v);model.zero_grad(set_to_none=True);value=0.
        ctx,y=train;N=len(y)
        for k in range(0,N,batch_size):
            loss=-model.log_probability(subset(ctx,slice(k,k+batch_size)),y[k:k+batch_size]).sum()/N
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite likelihood')
            loss.backward();value+=float(loss.detach())
        penalized=[p.square().sum() for n,p in parameters if n not in ('lam','kappa','rho','count_offset')]
        if penalized:
            penalty=ridge*torch.stack(penalized).sum()/(2*N)
            penalty.backward();value+=float(penalty.detach())
        gradient=np.concatenate([p.grad.detach().numpy().ravel() for _,p in parameters])
        if not np.isfinite(gradient).all():raise RuntimeError('Nonfinite gradient')
        gradient=gradient*scales
        last_evaluation.update(vector=v.copy(),gradient=gradient.copy())
        return value,gradient
    @torch.no_grad()
    def score(data):
        ctx,y=data
        return float(torch.cat([model.log_probability(subset(ctx,slice(k,k+batch_size)),y[k:k+batch_size])
                                 for k in range(0,len(y),batch_size)]).mean())
    write_json(out/'config.json',model.config)
    write_json(out/'protocol.json',{'optimizer':'L-BFGS-B','maxiter':maxiter,'ridge':ridge,
        'maxcalls':maxcalls,'time_limit':time_limit,
        'batch_size':batch_size,'seed':model.config.get('seed'),'stationarity_threshold':threshold,
        'lbfgs_history':history_size,'coordinate_preconditioning':'fixed approximate diagonal Fisher; identical raw objective/bounds/gradient certificate',
        'selection':'maximum validation basket log likelihood, including initialization',
        'model':model.config.get('model_family','separated-price category chain')+'; fully offered, nonempty distinct-item baskets',
        'train_baskets':len(train[1]),'validation_baskets':len(validation[1]),
        'torch':torch.__version__,'numpy':np.__version__})
    initial=vector();initial_objective,_=objective(initial)
    best={'validation_ll':score(validation),'iteration':0};history=[];start=time.monotonic()
    torch.save(model.state_dict(),out/'best.pt')
    last_accepted=initial.copy();calls=0
    def callback(v):
        nonlocal last_accepted
        last_accepted=v.copy()
        assign(v);vl=score(validation)
        row={'iteration':len(history)+1,'validation_ll':vl,'seconds':time.monotonic()-start}
        if np.array_equal(last_evaluation.get('vector'),v):
            row['original_projected_gradient']=gradient_norm(v,last_evaluation['gradient'],bounds)
            row['scaled_projected_gradient']=raw_norm(v,last_evaluation['gradient'],bounds)
        history.append(row);write_json(out/'history.json',history)
        torch.save(model.state_dict(),out/'latest.pt')
        if vl>best['validation_ll']:
            best.update(validation_ll=vl,iteration=row['iteration']);torch.save(model.state_dict(),out/'best.pt')
        print(json.dumps(row),flush=True)
        if row.get('original_projected_gradient',float('inf'))<=threshold and row.get('scaled_projected_gradient',float('inf'))<=threshold:raise StopIteration
    class FitBudget(Exception):pass
    def limited(v):
        nonlocal calls
        if maxcalls is not None and calls>=maxcalls:raise FitBudget('objective_call_limit')
        if time_limit is not None and time.monotonic()-start>=time_limit:raise FitBudget('time_limit')
        calls+=1
        return objective(v)
    try:
        opt=minimize(limited,initial,jac=True,method='L-BFGS-B',bounds=bounds,callback=callback,
                     options={'maxiter':maxiter,'maxfun':maxiter*5,'gtol':threshold/10,'ftol':1e-12,'maxls':30,'maxcor':history_size})
    except FitBudget as exc:
        from types import SimpleNamespace
        opt=SimpleNamespace(x=last_accepted,nit=len(history),nfev=calls,message=str(exc))
    final_objective,final_grad=objective(opt.x)
    final_stationarity=gradient_norm(opt.x,final_grad,bounds)
    torch.save(model.state_dict(),out/'final.pt')
    final_validation=score(validation)
    # Evaluate the predictive checkpoint's stationarity separately from the final iterate.
    model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=True))
    best_vector=vector();best_objective,best_grad=objective(best_vector)
    best_stationarity=gradient_norm(best_vector,best_grad,bounds)
    result={'initial_objective':initial_objective,'final_objective':final_objective,
        'best_objective':best_objective,'best_validation_ll':best['validation_ll'],
        'final_validation_ll':final_validation,'best_iteration':best['iteration'],
        'iterations':int(opt.nit),'calls':int(opt.nfev),'optimizer_message':str(opt.message),
        'best_projected_gradient':best_stationarity,'final_projected_gradient':final_stationarity,
        'best_stationarity_passed':best_stationarity<=threshold,
        'final_stationarity_passed':final_stationarity<=threshold and raw_norm(opt.x,final_grad,bounds)<=threshold,
        'final_original_stationarity_passed':final_stationarity<=threshold,
        'final_scaled_projected_gradient':raw_norm(opt.x,final_grad,bounds),
        'final_scaled_stationarity_passed':raw_norm(opt.x,final_grad,bounds)<=threshold,
        'best_scaled_projected_gradient':raw_norm(best_vector,best_grad,bounds),
        'parameter_count':len(initial),'seconds':time.monotonic()-start}
    write_json(out/'fit.json',result)
    return result
