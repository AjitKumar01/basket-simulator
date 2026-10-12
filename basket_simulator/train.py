"""Train from context/member NPZ arrays using bounded full-batch L-BFGS-B."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import torch
from .trainable import TrainableBasketModel

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


def gradient_norm(vector,gradient,bounds):
    g=gradient.copy()
    for i,(lo,hi) in enumerate(bounds):
        if lo is not None and vector[i]<=lo+1e-9:g[i]=min(g[i],0.)
        if hi is not None and vector[i]>=hi-1e-9:g[i]=max(g[i],0.)
    return float(abs(g).max())


def fit(model,train,validation,out,maxiter=200,ridge=10.,batch_size=512,threshold=2e-4,
        maxcalls=None,time_limit=None):
    """Write best/final states and diagnostics. Existing run directories are rejected."""
    if maxiter<1 or ridge<0 or batch_size<1 or threshold<=0:raise ValueError('Invalid fit settings')
    if (maxcalls is not None and maxcalls<1) or (time_limit is not None and time_limit<=0):
        raise ValueError('Fit budgets must be positive')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad];bounds=model.parameter_bounds()
    if not parameters:raise ValueError('No trainable parameters')
    def vector():return np.concatenate([p.detach().numpy().ravel() for _,p in parameters])
    def assign(v):
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
        penalized=[p.square().sum() for n,p in parameters if n not in ('lam','kappa','rho')]
        if penalized:
            penalty=ridge*torch.stack(penalized).sum()/(2*N)
            penalty.backward();value+=float(penalty.detach())
        gradient=np.concatenate([p.grad.detach().numpy().ravel() for _,p in parameters])
        if not np.isfinite(gradient).all():raise RuntimeError('Nonfinite gradient')
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
        'selection':'maximum validation basket log likelihood, including initialization',
        'model':'separated-price category chain; fully offered, nonempty distinct-item baskets',
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
        history.append(row);write_json(out/'history.json',history)
        torch.save(model.state_dict(),out/'latest.pt')
        if vl>best['validation_ll']:
            best.update(validation_ll=vl,iteration=row['iteration']);torch.save(model.state_dict(),out/'best.pt')
        print(json.dumps(row),flush=True)
    class FitBudget(Exception):pass
    def limited(v):
        nonlocal calls
        if maxcalls is not None and calls>=maxcalls:raise FitBudget('objective_call_limit')
        if time_limit is not None and time.monotonic()-start>=time_limit:raise FitBudget('time_limit')
        calls+=1
        return objective(v)
    try:
        opt=minimize(limited,initial,jac=True,method='L-BFGS-B',bounds=bounds,callback=callback,
                     options={'maxiter':maxiter,'maxfun':maxiter*5,'gtol':threshold/10,'ftol':1e-12,'maxls':30})
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
        'final_stationarity_passed':final_stationarity<=threshold,
        'parameter_count':len(initial),'seconds':time.monotonic()-start}
    write_json(out/'fit.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--maxiter',type=int,default=200)
    p.add_argument('--ridge',type=float,default=10.);p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--init-checkpoint',help='Warm start parameter values, not optimizer history')
    args=p.parse_args()
    if args.threads<1:p.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    manifest={'files':{
        str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (args.train,args.validation,args.config,*([args.init_checkpoint] if args.init_checkpoint else []))},
        'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in Path(__file__).parent.glob('*.py')}}
    config=json.loads(Path(args.config).read_text());config['seed']=args.seed
    train=load_data(args.train,config);validation=load_data(args.validation,config)
    model=(TrainableBasketModel.from_checkpoint(args.init_checkpoint,args.config)
           if args.init_checkpoint else TrainableBasketModel(config,args.seed))
    model.config['seed']=args.seed
    result=fit(model,train,validation,args.out,args.maxiter,args.ridge,args.batch_size)
    write_json(Path(args.out)/'input_manifest.json',manifest)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
