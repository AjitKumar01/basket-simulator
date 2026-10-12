"""Exhaustive checks independent of research data and adjacent repositories."""
import itertools
import unittest
import numpy as np
from basket_simulator.model import BasketSimulator, toy_model


class BasketSimulatorTests(unittest.TestCase):
    def setUp(self):
        self.model,self.context = toy_model()

    def test_exhaustive_normalization_and_marginals(self):
        baskets = np.array(list(itertools.product((0,1),repeat=6)))[1:].reshape(-1,2,3)
        mass = np.exp([self.model.log_probability(self.context,y) for y in baskets])
        self.assertAlmostEqual(mass.sum(),1.,places=13)
        exact = np.einsum('b,bcj->cj',mass,baskets)
        np.testing.assert_allclose(exact,self.model.predict(self.context)['incidence'],atol=1e-13)
        self.assertAlmostEqual((mass*baskets.sum((1,2))).sum(),
                               self.model.predict(self.context)['expected_distinct_size'],places=13)

    def test_sampling_support_reproducibility_and_moments(self):
        y = self.model.generate(self.context,50000,92)
        self.assertTrue(y.any((1,2)).all())
        np.testing.assert_array_equal(y,self.model.generate(self.context,50000,92))
        p = self.model.predict(self.context)['incidence']
        # Six marginal errors tested against conservative Monte Carlo standard errors.
        self.assertTrue((abs(y.mean(0)-p)<6*np.sqrt(p*(1-p)/len(y))).all())

    def test_action_reference_and_input_immutability(self):
        before = np.array(self.context['dlp']).copy()
        change = self.model.what_if(self.context,0,.9)
        altered = {**self.context,'dlp':before.copy()}
        altered['dlp'].flat[0] += np.log(.9)
        np.testing.assert_allclose(change['after']['incidence'],self.model.predict(altered)['incidence'])
        np.testing.assert_array_equal(before,self.context['dlp'])
        nochange = self.model.what_if(self.context,0,1.)
        np.testing.assert_array_equal(nochange['incidence_change'],np.zeros((2,3)))

    def test_targets_and_unsupported_extensions_rejected(self):
        with self.assertRaises(ValueError):
            self.model.predict({**self.context,'member':np.ones((2,3))})
        with self.assertRaises(ValueError):
            BasketSimulator({**self.model.s,'gamma':np.ones(1)},self.model.config)
        with self.assertRaises(ValueError):
            self.model.what_if(self.context,-1,.9)

    def test_exact_completion_scoring(self):
        revealed = np.zeros((2,3));revealed[0,0]=1
        order,scores = self.model.recommend(self.context,revealed)
        self.assertNotIn(0,order)
        for j in order:
            y = revealed.copy();y.flat[j]=1
            self.assertAlmostEqual(scores[j],self.model.log_probability(self.context,y),places=14)
        self.assertTrue((np.diff(scores[order])<=0).all())


if __name__=='__main__':unittest.main()
