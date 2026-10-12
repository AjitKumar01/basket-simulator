"""Standalone differentiable separated-price basket chain, in float64.

The law matches the current research model; no neighboring repository is imported.
Uniform category width and a fully offered catalogue are explicit constraints.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def log_esp(u, maximum):
    """Log elementary symmetric polynomials, including E_0=1; O(P*R)."""
    terms = [torch.zeros_like(u[...,0])]+[torch.full_like(u[...,0],-1e30) for _ in range(maximum)]
    for j in range(u.shape[-1]):
        terms = [terms[0]]+[torch.logaddexp(terms[n],u[...,j]+terms[n-1])
                               if n<=j+1 else terms[n] for n in range(1,maximum+1)]
    return torch.stack(terms,-1)


class TrainableBasketModel(nn.Module):
    def __init__(self, config, seed=0):
        super().__init__()
        self.config = dict(config)
        C,P,R = (int(config[k]) for k in ('C','P','R'))
        if not 1<=R<=P or C<1 or P<1:
            raise ValueError('Require C>=1 and 1<=R<=P')
        self.C,self.P,self.R = C,P,R
        J = C*P
        users,stores = int(config['users']),int(config['stores'])
        if min(users,stores)<1:raise ValueError('users/stores include at least unknown row0')
        taste,store_rank,hidden,cross = (int(config.get(k,d)) for k,d in
            (('taste_rank',4),('store_rank',2),('hidden',8),('cross_rank',min(C,4))))
        if min(taste,store_rank,hidden,cross)<1:raise ValueError('Ranks must be positive')
        g = torch.Generator().manual_seed(seed)
        def small(*shape):return nn.Parameter(.01*torch.randn(*shape,generator=g,dtype=torch.float64))
        def zero(*shape):return nn.Parameter(torch.zeros(*shape,dtype=torch.float64))
        self.lam=zero(J);self.alpha=small(J,taste);self.theta=small(users,taste)
        self.zeta=small(J,store_rank);self.xi=small(stores,store_rank);self.mu=small(J,4)
        self.w_dsp=zero(J);self.w_mlr=zero(J)
        self.rho=nn.Parameter(torch.full((C,),.5,dtype=torch.float64))
        self.choice_price=nn.Parameter(torch.full((J,),.5,dtype=torch.float64))
        self.category_price=nn.Parameter(torch.full((C,P),.5/P,dtype=torch.float64))
        self.count_price_scale=nn.Parameter(torch.full((C,),.5,dtype=torch.float64))
        self.kappa=zero(C);self.tau=nn.Parameter(torch.full((C,),math.log(math.expm1(1.)),dtype=torch.float64))
        self.cross_p=small(C,cross);self.cross_q=small(C,cross)
        self.V=zero(C,C);self.W=small(C,hidden);self.h0=zero(hidden);self.omega=zero(C,hidden)
        self.U=zero(C,taste)
        self.register_buffer('psi0',torch.zeros(C+1,dtype=torch.float64))
        self.register_buffer('psi1',torch.zeros(C,dtype=torch.float64))
        self.register_buffer('layout',torch.arange(J).reshape(C,P))
        order=torch.tensor(config.get('order',list(range(C))),dtype=torch.long)
        if not torch.equal(order.sort().values,torch.arange(C)):raise ValueError('order must be a permutation')
        self.register_buffer('order',order)
        support=torch.tensor(config['calendar_support'],dtype=torch.float64)
        if support.shape!=(2,) or not torch.isfinite(support).all() or support[0]>support[1]:
            raise ValueError('calendar_support requires two finite ordered endpoints')
        self.register_buffer('calendar_support',support)
        self.register_buffer('separate_category_price',torch.tensor(True))
        with torch.no_grad():self.theta[0].zero_();self.xi[0].zero_()

    @classmethod
    def from_checkpoint(cls, checkpoint, config):
        from .model import BasketSimulator
        reference=BasketSimulator.load(checkpoint,config)
        if not bool(reference.s['separate_category_price']):
            raise ValueError('Trainable release implements the current separated law only')
        cfg=dict(reference.config)
        cfg.update(taste_rank=reference.s['alpha'].shape[1],store_rank=reference.s['zeta'].shape[1],
                   hidden=reference.s['W'].shape[1],cross_rank=reference.s['cross_p'].shape[1],
                   order=reference.s['order'].tolist())
        model=cls(cfg)
        # Inactive historical tensors were validated by the reference adapter above.
        model.load_state_dict({k:torch.as_tensor(reference.s[k]) for k in model.state_dict()})
        return model

    def baseline(self,ctx):
        week=ctx['week'].clamp(self.calendar_support[0],self.calendar_support[1])
        phase=(week-1)*2*np.pi/52
        cal=torch.stack((phase.sin(),phase.cos(),(2*phase).sin(),(2*phase).cos()),-1)
        b=(self.lam+self.theta[ctx['house']]@self.alpha.T
           +self.xi[ctx['store']]@self.zeta.T+cal@self.mu.T).reshape(-1,self.C,self.P)
        return b+self.w_dsp.reshape(self.C,self.P)*ctx['disp']+self.w_mlr.reshape(self.C,self.P)*ctx['mail']

    def stage_terms(self,ctx):
        b=self.baseline(ctx)
        u=b-self.choice_price.reshape(self.C,self.P)*ctx['dlp']
        actual=log_esp(u,self.R)
        reference=log_esp(b,self.R)
        n=torch.arange(1,self.R+1,dtype=torch.float64,device=b.device)
        penalty=self.rho[:,None]*(n*(n-1)/2)
        a0=reference[...,1:]-penalty
        a=a0+self.count_price_scale[None,:,None]*(actual[...,1:]-reference[...,1:])
        iv=torch.logsumexp(a0,-1)-(self.category_price*ctx['dlp']).sum(-1)
        cross=self.cross_p@self.cross_q.T
        cross=cross-torch.diag(torch.diag(cross))
        base=self.kappa+F.softplus(self.tau)*iv+iv@cross.T+self.theta[ctx['house']]@self.U.T
        return u,actual,F.log_softmax(a,-1),base

    def category_logits(self,base,x):
        ordered=x[...,self.order]
        contrib=ordered[...,None]*self.W
        prefix=torch.cumsum(contrib,-2)-contrib
        hidden=(F.relu(prefix+self.h0)*self.omega).sum(-1)
        counts=torch.cumsum(ordered,-1)-ordered
        return (base[...,self.order]+ordered@torch.tril(self.V,-1).T+hidden
                +self.psi0[counts.long()]+self.psi1*counts)

    def log_probability(self,ctx,member):
        u,esp,count,base=self.stage_terms(ctx)
        n=member.sum(-1).long();x=(n>0).to(u.dtype)
        if (n>self.R).any() or not x.any(-1).all():raise ValueError('Nonempty supported baskets required')
        eta=self.category_logits(base,x)
        ordered=x[:,self.order]
        unconditioned=(ordered*F.logsigmoid(eta)+(1-ordered)*F.logsigmoid(-eta)).sum(-1)
        eta0=self.category_logits(base,torch.zeros_like(x))
        log_empty=F.logsigmoid(-eta0).sum(-1)
        cat=unconditioned-torch.log(-torch.expm1(log_empty))
        within=((u*member).sum(-1)-esp.gather(-1,n[...,None]).squeeze(-1)
                +count.gather(-1,(n-1).clamp(min=0)[...,None]).squeeze(-1))
        return cat+(within*x).sum(-1)

    def reference(self):
        """Use the same fitted parameters with the portable application API."""
        from .model import BasketSimulator
        return BasketSimulator({k:v.detach().cpu().numpy() for k,v in self.state_dict().items()},self.config)

    def parameter_bounds(self):
        result=[]
        for name,p in self.named_parameters():
            bound=((0.,8.001) if name in ('choice_price','category_price') else
                   (0.,None) if name=='count_price_scale' else (-5.,10.) if name=='rho' else
                   (-10.,8.) if name=='tau' else (-15.,15.))
            result.extend([bound]*p.numel())
        return result
