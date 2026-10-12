"""Generate a small independent procedural dataset for exercising training.

This is a software example, not a real dataset or a causal robustness benchmark.
Observed covariates are random; category needs are independent conditional on
nonempty; products are drawn without replacement with price-sensitive weights.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def generate(out,train_size=1500,validation_size=400,seed=4):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(seed);C,P,users,stores=3,4,9,3
    config={'C':C,'P':P,'R':P,'users':users,'stores':stores,'calendar_support':[1,30],
            'products':[f'category_{c}:product_{j}' for c in range(C) for j in range(P)],
            'categories':[f'category_{c}' for c in range(C)],'scope':'procedural example only'}
    appeal=rng.normal(0,.4,(users,C,P))
    for name,N in (('train',train_size),('validation',validation_size)):
        house=rng.integers(1,users,N);store=rng.integers(1,stores,N)
        week=rng.integers(1,31,N)
        dlp=rng.uniform(-.25,.25,(N,C,P));disp=(rng.random((N,C,P))<.1).astype(float)
        mail=(rng.random((N,C,P))<.1).astype(float);member=np.zeros((N,C,P))
        for i in range(N):
            # Need decisions are independent of prices: discounts mostly switch brands.
            present=rng.random(C)<np.array([.45,.35,.25])
            while not present.any():present=rng.random(C)<np.array([.45,.35,.25])
            for c in np.flatnonzero(present):
                n=1+int(rng.random()<.1)
                weights=np.exp(appeal[house[i],c]-1.4*dlp[i,c]+.3*disp[i,c]+.2*mail[i,c])
                selected=rng.choice(P,n,replace=False,p=weights/weights.sum())
                member[i,c,selected]=1
        np.savez_compressed(out/f'{name}.npz',house=house,store=store,week=week,
                            dlp=dlp,disp=disp,mail=mail,member=member)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    context={'house':1,'store':1,'week':12,
             **{k:np.zeros((C,P)).tolist() for k in ('dlp','disp','mail')}}
    (out/'context.json').write_text(json.dumps(context,indent=2)+'\n')
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--train-size',type=int,default=1500)
    p.add_argument('--validation-size',type=int,default=400);p.add_argument('--seed',type=int,default=4)
    a=p.parse_args()
    if min(a.train_size,a.validation_size)<1:p.error('sizes must be positive')
    generate(a.out,a.train_size,a.validation_size,a.seed)
    print('Wrote config.json, context.json, train.npz and validation.npz to',a.out)


if __name__=='__main__':main()
