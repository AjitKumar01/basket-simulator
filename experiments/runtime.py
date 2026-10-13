"""Install the current verified numerical adapters without changing the model law."""
import numpy as np
import torch
from . import common, esp, optimizer, precondition
import basket_simulator.trainable as trainable
import basket_simulator.product_interactions as product


def install():
    generator = torch.Generator().manual_seed(972099)
    checks = []
    for width, degree, spread in [(2,2,1),(16,16,1),(16,16,100),(32,4,10)]:
        utility = (spread*torch.randn(3,4,width,generator=generator,dtype=torch.float64)).requires_grad_()
        a = esp.vector_log_esp(utility,degree)
        b = esp.fast_log_esp(utility,degree)
        ga = torch.autograd.grad(a[...,1:].sum(),utility,retain_graph=True)[0]
        gb = torch.autograd.grad(b[...,1:].sum(),utility)[0]
        np.testing.assert_allclose(a.detach(),b.detach(),atol=2e-12)
        np.testing.assert_allclose(ga,gb,atol=2e-12)
        checks.append({'width':width,'degree':degree,'spread':spread,'passed':True})
    common.write('numerical_adapter_checks.json',{'state':'passed','cases':checks})
    common.log_esp = trainable.log_esp = product.log_esp = common.historical_models.log_esp = esp.fast_log_esp

    def fit(model,train,validation,out,**kwargs):
        scales = precondition.scales(model,train,kwargs['ridge']) if isinstance(model,common.Independent) else None
        return optimizer.fit(model,train,validation,out,scales=scales,**kwargs)

    common.fit = fit
