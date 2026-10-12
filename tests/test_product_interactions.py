import json
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from basket_simulator.trainable import TrainableBasketModel
from basket_simulator.product_interactions import ProductInteractionModel
from basket_simulator.model import BasketSimulator
from basket_simulator.population import PopulationBasketSimulator


def fixture():
    cfg={'C':3,'P':2,'R':2,'users':2,'stores':1,'calendar_support':[1,52],
         'preference_mode':'direct','independent_only':True,'learn_inclusive_price':True}
    parent=TrainableBasketModel(cfg,17)
    with torch.no_grad():
        parent.lam.fill_(-1);parent.mu.zero_();parent.rho.zero_();parent.kappa.zero_()
        parent.choice_price.fill_(1);parent.count_price_scale.fill_(1);parent.inclusive_price_weight.fill_(1)
        parent.category_price.zero_();parent.cross_p.zero_();parent.tau.fill_(math.log(math.expm1(1)))
    context={'house':1,'store':0,'week':20,'dlp':np.zeros((3,2)),'disp':np.zeros((3,2)),'mail':np.zeros((3,2))}
    return ProductInteractionModel(parent,rank=2,seed=18),context


class ProductInteractionTests(unittest.TestCase):
    def test_uncertified_improvement_cannot_be_exported(self):
        from basket_simulator.pipeline import product_stage_decision
        self.assertIsNone(product_stage_decision({'best_validation_ll':-2.,'final_stationarity_passed':False},-3.))
        self.assertFalse(product_stage_decision({'best_validation_ll':-4.,'final_stationarity_passed':False},-3.))
        self.assertFalse(product_stage_decision({'best_validation_ll':-3.+1e-12,'final_stationarity_passed':True},-3.))
        self.assertTrue(product_stage_decision({'best_validation_ll':-2.,'final_stationarity_passed':True},-3.))

    def test_capped_preference_parent_exact_partition(self):
        from basket_simulator.pipeline import independent_start
        cfg={'C':2,'P':3,'R':2,'users':2,'stores':1,'calendar_support':[1,52]}
        y=((np.arange(1,64)[:,None]>>np.arange(6))&1).reshape(-1,2,3)
        y=y[(y.sum(-1)<=2).all(-1)];self.assertEqual(len(y),48)
        target=torch.tensor(y,dtype=torch.float64);parent=independent_start(cfg,target,31)
        ctx={'house':torch.ones(48,dtype=torch.long),'store':torch.zeros(48,dtype=torch.long),'week':torch.full((48,),20.,dtype=torch.float64),**{k:torch.zeros((48,2,3),dtype=torch.float64) for k in ('dlp','disp','mail')}}
        ll=parent.log_probability(ctx,target).detach().numpy()
        np.testing.assert_allclose(ll,-np.log(48),atol=1e-6)
        self.assertAlmostEqual(np.exp(ll).sum(),1,places=12)
        head=ProductInteractionModel(parent,rank=2,seed=32)
        np.testing.assert_allclose(head.log_probability(ctx,target).detach(),ll,atol=1e-12)

    def test_normalization_parent_nesting_and_sampler(self):
        torch.set_num_threads(1);m,c=fixture();s=m.reference()
        y=((np.arange(1,64)[:,None]>>np.arange(6))&1).reshape(-1,3,2)
        ctx=s._context(c,len(y));target=torch.tensor(y,dtype=torch.float64)
        np.testing.assert_allclose(m.log_probability(ctx,target).detach(),m.parent.log_probability(ctx,target).detach(),atol=1e-12)
        with torch.no_grad():m.A[0,0,0]=1.5;m.B[1,0,0]=1.5
        p=m.log_probability(ctx,target).detach().exp().numpy();self.assertAlmostEqual(p.sum(),1,places=12)
        loss=-m.log_probability(ctx,target).mean();loss.backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters() if v.requires_grad))
        sample=s.generate(c,20000,19);truth=(p@y.reshape(-1,6)).reshape(3,2)
        se=np.sqrt(truth*(1-truth)/len(sample));self.assertLess(float((abs(sample.mean(0)-truth)/se).max()),5)
        self.assertTrue(sample.any((1,2)).all())
        eligible=y.any(-1).all(-1);q=p[eligible]/p[eligible].sum();items=y[eligible].reshape(-1,6)
        covariance=q@(items[:,0]*items[:,2])-(q@items[:,0])*(q@items[:,2])
        self.assertGreater(covariance,.005)

    def test_load_applications_and_population(self):
        m,c=fixture()
        with torch.no_grad():m.A[0,0,0]=1;m.B[1,0,0]=1
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);torch.save(m.state_dict(),root/'best.pt');(root/'config.json').write_text(json.dumps(m.config))
            s=BasketSimulator.load(root/'best.pt',root/'config.json');basket=s.generate(c,1,20)[0]
            self.assertAlmostEqual(s.log_probability(c,basket),m.reference().log_probability(c,basket),places=12)
            restored=TrainableBasketModel.from_checkpoint(root/'best.pt',root/'config.json')
            self.assertIsInstance(restored,ProductInteractionModel)
            self.assertAlmostEqual(restored.reference().log_probability(c,basket),s.log_probability(c,basket),places=12)
            exact=s.predict(c,method='exact');mc=s.predict(c,draws=20000,seed=20)
            np.testing.assert_allclose(exact['incidence'],mc['incidence'],atol=.02)
            change=s.what_if(c,0,.9,draws=4096,seed=21)
            self.assertEqual(change['incidence_change'].shape,(3,2));self.assertIn('size_change_standard_error',change)
            self.assertGreater(change['incidence_change'][0,0],0)
            pop=PopulationBasketSimulator(s,[1],[1]);unknown={**c,'house':0}
            self.assertTrue(pop.generate(unknown,16,22).any((1,2)).all())
            order,score=s.recommend(c,draws=512,seed=22);self.assertEqual(len(order),6)

    def test_large_category_path_has_no_dense_category_matrix(self):
        cfg={'C':64,'P':2,'R':2,'users':2,'stores':1,'calendar_support':[1,52],
             'preference_mode':'factors','chain_rank':4}
        m=ProductInteractionModel(TrainableBasketModel(cfg,23),rank=4,seed=24)
        self.assertFalse(hasattr(m,'category'));self.assertEqual(m.category_p.numel(),256)
        self.assertLessEqual(len(m.orders),4)
        c={'house':0,'store':0,'week':20,'dlp':np.zeros((64,2)),'disp':np.zeros((64,2)),'mail':np.zeros((64,2))}
        ctx=m.reference()._context(c,2);y=m.sample(ctx,25).double();loss=-m.log_probability(ctx,y).mean();loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters() if v.requires_grad))


if __name__=='__main__':unittest.main()
