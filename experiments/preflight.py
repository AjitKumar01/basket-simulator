import itertools,time
from . import common as c
import torch,numpy as np
from torch.nn import functional as F

def original(u,R):
 t=[torch.zeros_like(u[...,0])]+[torch.full_like(u[...,0],-1e30) for _ in range(R)]
 for j in range(u.shape[-1]):t=[t[0]]+[torch.logaddexp(t[n],u[...,j]+t[n-1]) if n<=j+1 else t[n] for n in range(1,R+1)]
 return torch.stack(t,-1)
def main():
 g=torch.Generator().manual_seed(972000);checks=[]
 for P,R in [(2,2),(5,3),(16,16),(32,4)]:
  u=torch.randn(3,4,P,generator=g,dtype=torch.float64,requires_grad=True);old=original(u,R);new=c.log_esp(u,R);np.testing.assert_allclose(old.detach(),new.detach(),atol=1e-13)
  a=torch.autograd.grad(old[...,1:].sum(),u,retain_graph=True)[0];b=torch.autograd.grad(new[...,1:].sum(),u)[0];np.testing.assert_allclose(a,b,atol=1e-12);checks.append({'P':P,'R':R,'values_and_gradients':'passed'})
 cfg={'C':3,'P':2,'R':2,'users':3,'stores':2,'calendar_support':[9,82],'chain_rank':0,'lbfgs_history':3}
 y=torch.tensor(((np.arange(1,64)[:,None]>>np.arange(6))&1).reshape(-1,3,2),dtype=torch.uint8);m=c.independent_start(cfg,y,972000);fast=c.Independent(m.config);fast.load_state_dict(m.state_dict())
 ctx={'house':torch.ones(63,dtype=torch.long),'store':torch.ones(63,dtype=torch.long),'week':torch.full((63,),40.,dtype=torch.float64),**{k:torch.randn(63,3,2,generator=g,dtype=torch.float64) for k in ('dlp','disp','mail')}}
 with torch.no_grad():m.house_effect.normal_(0,.1);m.choice_price.fill_(1);fast.load_state_dict(m.state_dict())
 a=m.log_probability(ctx,y);b=fast.log_probability(ctx,y);np.testing.assert_allclose(a.detach(),b.detach(),atol=1e-12);a.sum().backward();b.sum().backward()
 for (n,p),(n2,q) in zip(m.named_parameters(),fast.named_parameters()):
  if p.requires_grad:np.testing.assert_allclose(p.grad,q.grad,atol=1e-11)
 repeated={k:v[:1].repeat(63,*([1]*(v.ndim-1))) for k,v in ctx.items()};prob=fast.log_probability(repeated,y).exp().detach().numpy();np.testing.assert_allclose(prob.sum(),1,atol=1e-12)
 drawn=fast.sample({k:v[:1].repeat(30000,*([1]*(v.ndim-1))) for k,v in ctx.items()},972001);truth=prob@y.reshape(-1,6).numpy();np.testing.assert_allclose(drawn.reshape(-1,6).double().mean(0),truth,atol=.014)
 cfg={**m.config,'independent_only':False,'trainable_parameters':['rho','kappa','V']};slow=c.TrainableBasketModel(cfg);slow.load_state_dict(m.state_dict());cached=c.CachedStructure(cfg);cached.load_state_dict(slow.state_dict())
 with torch.no_grad():slow.rho.fill_(.7);slow.kappa.fill_(.2);slow.V.normal_(0,.1);cached.load_state_dict(slow.state_dict());base=slow.baseline(ctx);u=base-slow.choice_price.reshape(3,2)*ctx['dlp'];cx={**ctx,'_u':u,'_actual':c.log_esp(u,2),'_reference':c.log_esp(base,2)}
 a=slow.log_probability(ctx,y);b=cached.log_probability(cx,y);np.testing.assert_allclose(a.detach(),b.detach(),atol=1e-12);a.sum().backward();b.sum().backward()
 for (n,p),(n2,q) in zip(slow.named_parameters(),cached.named_parameters()):
  if p.requires_grad:np.testing.assert_allclose(p.grad,q.grad,atol=1e-11)
 report={'state':'passed','ESP':checks,'independent_likelihood_gradient_normalization_sampling':'passed','cached_structure_likelihood_gradient':'passed'}
 if (c.OUT/'data/config.json').exists():
  data=c.Data('train');cfg=c.json.loads((c.OUT/'data/config.json').read_text());m=c.independent_start(cfg,data.y,972301);fast=c.Independent(m.config);fast.load_state_dict(m.state_dict());x,y=data.batch(slice(0,512));t=time.monotonic();loss=-fast.log_probability(x,y).mean();loss.backward();report['pilot']={'independent_512_forward_backward_seconds':time.monotonic()-t,'rows':len(data),'parameters':sum(p.numel() for p in fast.parameters() if p.requires_grad)}
 c.write('preflight.json',report);print(report)
if __name__=='__main__':main()
