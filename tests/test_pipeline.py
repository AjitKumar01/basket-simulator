import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
from basket_simulator.example_data import generate
from basket_simulator.pipeline import staged_fit
from basket_simulator.train import load_data
from basket_simulator.trainable import TrainableBasketModel


class PipelineTests(unittest.TestCase):
    def test_staged_training_and_parent_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=generate(root/'data',120,40,91)
            train=load_data(root/'data/train.npz',config)
            val=load_data(root/'data/validation.npz',config)
            with contextlib.redirect_stdout(io.StringIO()):
                model,result=staged_fit(config,train,val,root/'fit',ridges=(1.,10.),
                                         maxiter=5,maxcalls=12,batch_size=64)
            self.assertGreaterEqual(result['structure_validation_ll_gain'],-1e-12)
            checkpoint=TrainableBasketModel.from_checkpoint(root/'fit/best.pt',root/'fit/config.json')
            np.testing.assert_allclose(model.log_probability(*val).detach(),checkpoint.log_probability(*val).detach(),atol=1e-12)
            self.assertIn('No joint optimum',result['stationarity_scope'])


if __name__=='__main__':unittest.main()
