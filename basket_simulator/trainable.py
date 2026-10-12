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
        chain_rank=int(config.get('chain_rank',4 if C>10 else 0))
        if chain_rank<0:raise ValueError('chain_rank must be nonnegative')
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
        self.V=zero(C,C) if not chain_rank else zero(0,0)
        self.W=small(C,hidden);self.h0=zero(hidden);self.omega=zero(C,hidden)
        self.U=zero(C,taste)
        mode=config.get('preference_mode','factors')
        if mode not in ('factors','direct'):raise ValueError('preference_mode must be factors or direct')
        if mode=='direct':
            self.house_effect=zero(users,J);self.store_effect=zero(stores,J)
            for name in ('alpha','theta','zeta','xi','U'):
                getattr(self,name).requires_grad_(False)
                with torch.no_grad():getattr(self,name).zero_()
        if config.get('learn_inclusive_price',False):
            self.inclusive_price_weight=nn.Parameter(torch.ones(C,dtype=torch.float64))
        if config.get('category_anchor','legacy')=='independent':
            self.register_buffer('independent_category_anchor',torch.tensor(True))
            if hasattr(self,'inclusive_price_weight'):self.inclusive_price_weight.requires_grad_(False)
        elif config.get('category_anchor','legacy')!='legacy':raise ValueError('Invalid category_anchor')
        if chain_rank:
            self.linear_p=zero(C,chain_rank);self.linear_q=small(C,chain_rank)
            self.V.requires_grad_(False)
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
        if config.get('independent_only',False):
            if mode!='direct':raise ValueError('independent_only requires direct preferences')
            for name,p in self.named_parameters():
                p.requires_grad_(name in ('lam','house_effect','store_effect','mu','choice_price','w_dsp','w_mlr'))
        if 'trainable_parameters' in config:
            known={n for n,_ in self.named_parameters()}
            requested=set(config['trainable_parameters'])
            if not requested<=known:raise ValueError('Unknown trainable parameter names')
            for name,p in self.named_parameters():p.requires_grad_(name in requested)

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
        cfg['chain_rank']=reference.s['linear_p'].shape[1] if 'linear_p' in reference.s else 0
        cfg['preference_mode']='direct' if 'house_effect' in reference.s else 'factors'
        cfg['learn_inclusive_price']='inclusive_price_weight' in reference.s
        cfg['category_anchor']='independent' if bool(reference.s.get('independent_category_anchor',False)) else 'legacy'
        model=cls(cfg)
        # Inactive historical tensors were validated by the reference adapter above.
        model.load_state_dict({k:torch.as_tensor(reference.s[k]) for k in model.state_dict()})
        return model

    @classmethod
    def from_independent(cls,checkpoint,config):
        """Exact nesting of a conditional Bernoulli fit; no target data consumed."""
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cfg={**config,'preference_mode':'direct','learn_inclusive_price':True}
        if cfg['R']!=cfg['P']:raise ValueError('Independent law requires full within-category count support')
        m=cls(cfg)
        with torch.no_grad():
            for name,key in (('lam','intercept'),('house_effect','house_effect'),('store_effect','store_effect'),
                             ('choice_price','price'),('w_dsp','display'),('w_mlr','advert')):
                getattr(m,name).copy_(state[key])
            m.mu.copy_(state['calendar'].T)
            m.rho.zero_();m.count_price_scale.fill_(1.);m.inclusive_price_weight.fill_(1.)
            m.category_price.zero_();m.kappa.zero_();m.tau.fill_(math.log(math.expm1(1.)))
            # One cross factor stays seeded: the zero interaction can acquire a gradient.
            m.cross_p.zero_();m.V.zero_();m.omega.zero_()
        return m

    def baseline(self,ctx):
        week=ctx['week'].clamp(self.calendar_support[0],self.calendar_support[1])
        phase=(week-1)*2*np.pi/52
        cal=torch.stack((phase.sin(),phase.cos(),(2*phase).sin(),(2*phase).cos()),-1)
        if hasattr(self,'house_effect'):
            b=self.lam+self.house_effect[ctx['house']]+self.store_effect[ctx['store']]+cal@self.mu.T
        else:
            b=self.lam+self.theta[ctx['house']]@self.alpha.T+self.xi[ctx['store']]@self.zeta.T+cal@self.mu.T
        b=b.reshape(-1,self.C,self.P)
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
        iv0=torch.logsumexp(a0,-1)
        weight=getattr(self,'inclusive_price_weight',torch.zeros_like(self.rho))
        if hasattr(self,'independent_category_anchor'):
            iv=torch.logsumexp(actual[...,1:],-1)-(self.category_price*ctx['dlp']).sum(-1)
        else:
            iv=iv0+weight*(torch.logsumexp(actual[...,1:]-penalty,-1)-iv0)-(self.category_price*ctx['dlp']).sum(-1)
        cross=(iv@self.cross_q)@self.cross_p.T-iv*(self.cross_p*self.cross_q).sum(-1)
        base=self.kappa+F.softplus(self.tau)*iv+cross+self.theta[ctx['house']]@self.U.T
        return u,actual,F.log_softmax(a,-1),base

    def category_logits(self,base,x):
        ordered=x[...,self.order]
        contrib=ordered[...,None]*self.W
        prefix=torch.cumsum(contrib,-2)-contrib
        hidden=(F.relu(prefix+self.h0)*self.omega).sum(-1)
        counts=torch.cumsum(ordered,-1)-ordered
        if hasattr(self,'linear_p'):
            contributions=ordered[...,None]*self.linear_q
            linear=((torch.cumsum(contributions,-2)-contributions)*self.linear_p).sum(-1)
        else:linear=ordered@torch.tril(self.V,-1).T
        return (base[...,self.order]+linear+hidden
                +self.psi0[counts.long()]+self.psi1*counts)

    def log_probability(self,ctx,member):
        u,esp,count,base=self.stage_terms(ctx)
        n=member.sum(-1).long();x=(n>0).to(u.dtype)
        if (n>self.R).any() or not x.any(-1).all():raise ValueError('Nonempty supported baskets required')
        eta=self.category_logits(base,x)
        ordered=x[:,self.order]
        unconditioned=(ordered*F.logsigmoid(eta)+(1-ordered)*F.logsigmoid(-eta)).sum(-1)
        eta0=self.category_logits(base,torch.zeros_like(x))
        log_no=F.logsigmoid(-eta0)
        prefix_no=torch.cumsum(log_no,-1)-log_no
        log_nonempty=torch.logsumexp(F.logsigmoid(eta0)+prefix_no,-1)
        cat=unconditioned-log_nonempty
        within=((u*member).sum(-1)-esp.gather(-1,n[...,None]).squeeze(-1)
                +count.gather(-1,(n-1).clamp(min=0)[...,None]).squeeze(-1))
        return cat+(within*x).sum(-1)

    def reference(self):
        """Use the same fitted parameters with the portable application API."""
        from .model import BasketSimulator
        return BasketSimulator({k:v.detach().cpu().numpy() for k,v in self.state_dict().items()},self.config)

    @torch.no_grad()
    def sample(self,ctx,seed=0):
        """One exact nonempty basket per context; polynomial batched sampling."""
        u,_,count,base=self.stage_terms(ctx);B=len(base);device=u.device
        generator=torch.Generator(device=device).manual_seed(seed)
        x=torch.zeros((B,self.C),dtype=torch.float64,device=device)
        empty=self.category_logits(base,x)
        suffix=[torch.full((B,),-torch.inf,dtype=u.dtype,device=device)]
        for i in range(self.C-1,-1,-1):
            suffix.append(torch.logaddexp(F.logsigmoid(empty[:,i]),F.logsigmoid(-empty[:,i])+suffix[-1]))
        suffix=list(reversed(suffix));hidden=self.h0.expand(B,-1).clone()
        counts=torch.zeros(B,dtype=torch.long,device=device)
        linear=torch.zeros((B,self.linear_p.shape[1]),dtype=u.dtype,device=device) if hasattr(self,'linear_p') else None
        for i,c in enumerate(self.order):
            link=linear@self.linear_p[i] if linear is not None else x[:,self.order[:i]]@self.V[i,:i]
            eta=base[:,c]+link+(F.relu(hidden)*self.omega[i]).sum(-1)+self.psi0[counts]+self.psi1[i]*counts
            probability=torch.where(counts>0,eta.sigmoid(),(F.logsigmoid(empty[:,i])-suffix[i]).exp())
            selected=torch.rand(B,generator=generator,device=device)<probability.clamp(0,1)
            x[:,c]=selected;counts+=selected;hidden+=selected[:,None]*self.W[i]
            if linear is not None:linear+=selected[:,None]*self.linear_q[i]
        required=torch.multinomial(count.exp().reshape(-1,self.R),1,generator=generator).reshape(B,self.C)+1
        required*=x.long()
        esp=torch.full((B,self.C,self.P+1,self.R+1),-1e30,dtype=u.dtype,device=device);esp[...,0]=0.
        for j in range(self.P-1,-1,-1):
            esp[:,:,j,1:]=torch.logaddexp(esp[:,:,j+1,1:],u[:,:,j,None]+esp[:,:,j+1,:-1])
        out=torch.zeros_like(u,dtype=torch.bool)
        for j in range(self.P):
            numerator=esp[:,:,j+1].gather(-1,(required-1).clamp(min=0)[...,None]).squeeze(-1)
            denominator=esp[:,:,j].gather(-1,required[...,None]).squeeze(-1)
            probability=torch.where(required>0,(u[:,:,j]+numerator-denominator).exp(),0.).clamp(0,1)
            selected=(torch.rand((B,self.C),generator=generator,device=device)<probability)|(required==self.P-j)
            out[:,:,j]=selected;required-=selected.long()
        if required.any() or not out.any(-1).any(-1).all():raise RuntimeError('Exact sampler support failure')
        return out

    def parameter_bounds(self):
        result=[]
        for name,p in self.named_parameters():
            if not p.requires_grad:continue
            bound=((0.,8.001) if name in ('choice_price','category_price') else
                   (0.,1.) if name=='inclusive_price_weight' else
                   (0.,None) if name=='count_price_scale' else (-5.,10.) if name=='rho' else
                   (-10.,8.) if name=='tau' else (-15.,15.))
            result.extend([bound]*p.numel())
        return result
