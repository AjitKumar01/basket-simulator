"""Fixed diagonal Fisher coordinates. Objective, bounds and raw gradient unchanged."""
import numpy as np,torch
from . import common as c
from .optimizer import fit

def scales(model,data,ridge):
 ctx,y=data;N=len(y);J=model.C*model.P;freq=y.numpy().mean(0).ravel();v=np.clip(freq*(1-freq),1e-7,None)
 house=np.bincount(ctx['house'].numpy(),minlength=model.config['users'])/N;store=np.bincount(ctx['store'].numpy(),minlength=model.config['stores'])/N
 weeks=ctx['week'].numpy().clip(*model.config['calendar_support']);phase=(weeks-1)*2*np.pi/52;cal=np.stack([np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)],-1);diag={'lam':v,'house_effect':house[:,None]*v,'store_effect':store[:,None]*v,'mu':v[:,None]*(cal**2).mean(0)}
 for name,field in [('choice_price','dlp'),('w_dsp','disp'),('w_mlr','mail')]:
  lazy=ctx[field];second=np.zeros(J)
  if isinstance(lazy,torch.Tensor):
   diag[name]=v*lazy.numpy().reshape(N,J).__pow__(2).mean(0);continue
  count=np.bincount(lazy.rows,minlength=len(lazy.values))
  for start in range(0,len(count),1024):
   x=np.asarray(lazy.values[start:start+1024],dtype=float).reshape(-1,J);second+=count[start:start+len(x)]@x**2
  diag[name]=v*second/N
 rows=[]
 for name,p in model.named_parameters():
  if not p.requires_grad:continue
  d=diag[name]+(0 if name=='lam' else ridge/N);s=np.clip(1/np.sqrt(np.maximum(d,1e-8)),.05,10000);rows.append(np.broadcast_to(s,p.shape).ravel())
 return np.concatenate(rows)
