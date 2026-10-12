import itertools
import unittest
import numpy as np
import torch
from basket_simulator.trainable import TrainableBasketModel


class CountOffsetTests(unittest.TestCase):
    def test_mean_and_dispersion_extension_preserves_law_and_gradient(self):
        torch.set_num_threads(1)
        cfg={'C':2,'P':3,'R':3,'users':2,'stores':2,'calendar_support':[1,30]}
        old=TrainableBasketModel(cfg,12)
        new=TrainableBasketModel({**cfg,'learn_count_offset':True},12)
        new.load_state_dict(old.state_dict(),strict=False)
        y=torch.tensor(list(itertools.product((0.,1.),repeat=6))[1:],dtype=torch.float64).reshape(-1,2,3)
        N=len(y);ctx={'house':torch.ones(N,dtype=torch.long),'store':torch.ones(N,dtype=torch.long),
                     'week':torch.full((N,),12.,dtype=torch.float64),
                     **{k:torch.zeros((N,2,3),dtype=torch.float64) for k in ('dlp','disp','mail')}}
        np.testing.assert_allclose(new.log_probability(ctx,y).detach(),old.log_probability(ctx,y).detach(),atol=1e-12)
        with torch.no_grad():new.count_offset.copy_(torch.tensor([1.7,-.8]))
        ll=new.log_probability(ctx,y)
        self.assertAlmostEqual(float(ll.exp().sum().detach()),1.,places=12)
        context={k:(int(v[0]) if k in ('house','store') else float(v[0]) if k=='week' else v[0].numpy()) for k,v in ctx.items()}
        reference=new.reference()
        np.testing.assert_allclose([reference.log_probability(context,b.numpy()) for b in y],ll.detach(),atol=1e-12)
        loss=-ll.mean();loss.backward();analytic=float(new.count_offset.grad[0])
        original=float(new.count_offset[0].detach());eps=1e-5
        with torch.no_grad():
            new.count_offset[0]=original+eps;plus=float(-new.log_probability(ctx,y).mean())
            new.count_offset[0]=original-eps;minus=float(-new.log_probability(ctx,y).mean())
            new.count_offset[0]=original
        self.assertAlmostEqual(analytic,(plus-minus)/(2*eps),places=8)
        generated=reference.generate(context,50000,11).mean(0)
        np.testing.assert_allclose(generated,reference.predict(context)['incidence'],atol=.012)


if __name__=='__main__':unittest.main()
