"""Latent-mission and autoregressive comparators used in current benchmarks."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from basket_simulator.trainable import log_esp

def frozen(parent):
    m=copy.deepcopy(parent)
    for p in m.parameters():p.requires_grad_(False)
    return m

def nonempty(logits):
    no=F.logsigmoid(-logits);prefix=no.cumsum(-1)-no
    return torch.logsumexp(F.logsigmoid(logits)+prefix,-1)

class BinaryAR(nn.Module):
    """Personalized conditional binary autoregression, exactly nonempty-conditioned."""
    def __init__(self,parent,neural=False,seed=0):
        super().__init__();self.parent=frozen(parent);self.J=parent.C*parent.P;self.C=parent.C;self.P=parent.P;self.neural=neural
        self.config={**parent.config,'benchmark_baseline':'conditional_nade_H16' if neural else 'pairwise_product_autoregression'}
        self.lam=nn.Parameter(torch.zeros(self.J,dtype=torch.float64))
        self.house=nn.Parameter(torch.zeros((parent.config['users'],self.J),dtype=torch.float64))
        self.price=nn.Parameter(parent.choice_price.detach().clone())
        self.cross_price=nn.Parameter(torch.zeros((self.J,self.J),dtype=torch.float64))
        if neural:
            gen=torch.Generator().manual_seed(seed)
            self.E=nn.Parameter(.05*torch.randn(self.J,16,generator=gen,dtype=torch.float64))
            self.Q=nn.Parameter(.05*torch.randn(self.J,16,generator=gen,dtype=torch.float64))
            self.O=nn.Parameter(torch.zeros((self.J,16),dtype=torch.float64))
        else:self.W=nn.Parameter(torch.zeros((self.J,self.J),dtype=torch.float64))
    def logits(self,ctx,y):
        d=ctx['dlp'].reshape(-1,self.J)
        u=self.parent.baseline(ctx).reshape(-1,self.J)+self.lam+self.house[ctx['house']]-self.price*d
        cross=self.cross_price-torch.diag(self.cross_price.diag());u=u+d@cross.T
        if self.neural:
            contributions=y[...,None]*self.E;prefix=contributions.cumsum(1)-contributions
            u=u+(torch.sigmoid(prefix+(d@self.Q)[:,None])*self.O).sum(-1)
        else:u=u+y@torch.tril(self.W,-1).T
        return u
    def log_probability(self,ctx,member):
        y=member.reshape(-1,self.J);eta=self.logits(ctx,y);empty=self.logits(ctx,torch.zeros_like(y))
        return (y*F.logsigmoid(eta)+(1-y)*F.logsigmoid(-eta)).sum(-1)-nonempty(empty)
    def parameter_bounds(self):
        return [((0.,8.001) if name=='price' else (-15.,15.)) for name,p in self.named_parameters() if p.requires_grad for _ in range(p.numel())]
    @torch.no_grad()
    def sample(self,ctx,seed=0):
        B=len(ctx['house']);y=torch.zeros((B,self.J),dtype=torch.float64);g=torch.Generator().manual_seed(seed)
        empty=self.logits(ctx,y);suffix=[torch.full((B,),-torch.inf,dtype=torch.float64)]
        for j in range(self.J-1,-1,-1):suffix.append(torch.logaddexp(F.logsigmoid(empty[:,j]),F.logsigmoid(-empty[:,j])+suffix[-1]))
        suffix=list(reversed(suffix));active=torch.zeros(B,dtype=torch.bool)
        for j in range(self.J):
            eta=self.logits(ctx,y)[:,j];q=torch.where(active,eta.sigmoid(),(F.logsigmoid(empty[:,j])-suffix[j]).exp())
            chosen=torch.rand(B,generator=g)<q.clamp(0,1);y[:,j]=chosen;active|=chosen
        assert active.all();return y.reshape(B,self.C,self.P)

class MissionESP(nn.Module):
    """K2 individually nonempty-conditioned independent-category ESP components."""
    def __init__(self,parent,seed=0):
        super().__init__();self.parent=frozen(parent);self.C=parent.C;self.P=parent.P;self.R=parent.R
        self.config={**parent.config,'benchmark_baseline':'latent_mission_esp_K2'}
        g=torch.Generator().manual_seed(seed)
        self.taste=nn.Parameter(.05*torch.randn((2,self.C,self.P),generator=g,dtype=torch.float64))
        self.category=nn.Parameter(.05*torch.randn((2,self.C),generator=g,dtype=torch.float64))
        self.nu=nn.Parameter(torch.zeros((2,self.C),dtype=torch.float64))
        self.rho=nn.Parameter(parent.rho.detach().repeat(2,1))
        self.tau=nn.Parameter(parent.tau.detach().repeat(2,1))
        self.weights=nn.Parameter(torch.zeros((parent.config['users'],2),dtype=torch.float64))
    def terms(self,ctx,m):
        b=self.parent.baseline(ctx)+self.taste[m];u=b-self.parent.choice_price.reshape(self.C,self.P)*ctx['dlp']
        esp=log_esp(u,self.R);ref=log_esp(b,self.R);n=torch.arange(1,self.R+1,dtype=torch.float64)
        nu=getattr(self.parent,'count_offset',torch.zeros(self.C,dtype=torch.float64))+self.nu[m]
        penalty=self.rho[m,:,None]*(n*(n-1)/2)-nu[:,None]*(n-1)
        a0=ref[...,1:]-penalty;a=a0+self.parent.count_price_scale[None,:,None]*(esp[...,1:]-ref[...,1:])
        iv0=torch.logsumexp(a0,-1);iv=iv0+self.parent.inclusive_price_weight*(torch.logsumexp(esp[...,1:]-penalty,-1)-iv0)-(self.parent.category_price*ctx['dlp']).sum(-1)
        base=self.parent.kappa+self.category[m]+F.softplus(self.tau[m])*iv
        return u,esp,F.log_softmax(a,-1),base
    def component_ll(self,ctx,y,m):
        u,e,count,base=self.terms(ctx,m);n=y.sum(-1).long();x=(n>0).to(u.dtype)
        cat=(x*F.logsigmoid(base)+(1-x)*F.logsigmoid(-base)).sum(-1)-nonempty(base)
        within=(u*y).sum(-1)-e.gather(-1,n[...,None]).squeeze(-1)+count.gather(-1,(n-1).clamp(min=0)[...,None]).squeeze(-1)
        return cat+(within*x).sum(-1)
    def log_probability(self,ctx,y):
        weights=F.log_softmax(self.weights[ctx['house']],-1)
        return torch.logsumexp(torch.stack([self.component_ll(ctx,y,m) for m in range(2)],-1)+weights,-1)
    def parameter_bounds(self):
        return [((-5.,10.) if name=='rho' else (-10.,8.) if name=='tau' else (-15.,15.))
                for name,p in self.named_parameters() if p.requires_grad for _ in range(p.numel())]
