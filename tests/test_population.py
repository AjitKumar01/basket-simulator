import itertools
import unittest
import numpy as np
import torch
from basket_simulator import PopulationBasketSimulator
from basket_simulator.trainable import TrainableBasketModel


class PopulationTests(unittest.TestCase):
    def test_mixture_is_normalized_and_averages_probabilities(self):
        cfg={'C':2,'P':3,'R':3,'users':3,'stores':2,'calendar_support':[1,30],'preference_mode':'direct'}
        m=TrainableBasketModel(cfg,3)
        with torch.no_grad():
            m.house_effect[1].fill_(2.);m.house_effect[2].fill_(-2.)
        ref=m.reference();pop=PopulationBasketSimulator(ref,[1,2],[.5,.5])
        context={'house':0,'store':1,'week':12,**{k:np.zeros((2,3)) for k in ('dlp','disp','mail')}}
        baskets=np.array(list(itertools.product((0,1),repeat=6)))[1:].reshape(-1,2,3)
        mass=np.exp([pop.log_probability(context,y) for y in baskets])
        self.assertAlmostEqual(float(mass.sum()),1.,places=12)
        expected=np.einsum('b,bcj->cj',mass,baskets)
        np.testing.assert_allclose(pop.predict(context)['incidence'],expected,atol=1e-12)
        self.assertGreater(float(abs(expected-ref.predict(context)['incidence']).max()),.01)
        draws=pop.generate(context,60000,7)
        self.assertTrue(draws.any((1,2)).all())
        np.testing.assert_allclose(draws.mean(0),expected,atol=.012)
        shift=pop.what_if(context,0,.9)
        self.assertAlmostEqual(float(shift['incidence_change'].sum()),shift['size_change'],places=12)
        prior=PopulationBasketSimulator.from_training(ref,np.array([1,1,2]),128,9)
        self.assertTrue(np.isin(prior.households,[1,2]).all())
        with self.assertRaises(ValueError):pop.predict({**context,'house':1})


if __name__=='__main__':unittest.main()
