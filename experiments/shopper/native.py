"""Pinned native myopic SHOPPER: same actual ERIM train/validation arrays."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2');os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1');os.environ.setdefault('TF_NUM_INTEROP_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Fit optional native myopic SHOPPER; exact small-basket validation only')
parser.add_argument('--run',type=Path,required=True)
args=parser.parse_args()
os.environ['BASKET_RUN_DIR']=str(args.run.resolve())
from . import adapter as sh
from .adapter import epoch_rows,weights
OUT=args.run.resolve()/'shopper_native'
DATA=args.run.resolve()/'data'

def main():
    cfg=json.loads((DATA/'config.json').read_text());tr=sh.arrays(dict(np.load(DATA/'train.npz')),cfg['calendar_support']);va=sh.arrays(dict(np.load(DATA/'validation.npz')),cfg['calendar_support'])
    OUT.mkdir(exist_ok=False)
    import importlib.metadata as im
    sh.write(OUT/'native_source_manifest.json',{
        'protocol':'Native sampled loss/Adam; myopic, not Bayesian/thinking-ahead SHOPPER; screening only',
        'choice_learn':im.version('choice-learn'), 'native_module_sha256':sh.digest(Path(__import__('inspect').getfile(sh.Shopper))),
        'numpy':np.__version__, 'tensorflow':im.version('tensorflow'),
        'inputs':{str(p):sh.digest(p) for p in [DATA/'train.npz',DATA/'validation.npz',DATA/'config.json',Path(__file__)]}})
    if max(tr['member_flat'].sum(-1).max(),va['member_flat'].sum(-1).max())>12:
        raise ValueError('Exact unordered scoring exceeds the recorded 12-item budget; no observations dropped. Large SHOPPER needs a separately defined generation benchmark.')
    summaries=[]
    for rank in (4,8):
        for seed in (961301,961302):
            path=OUT/f'shopper_rank{rank}_{seed}'
            if (path/'fit.json').exists():summaries.append(json.loads((path/'fit.json').read_text()));continue
            path.mkdir();model=sh.make(rank,seed,cfg,tr,va);model.n_negative_samples=1;model.batch_size=128
            if (tr['member_flat'].sum(-1)>=cfg['C']*cfg['P']).any():raise ValueError('Native negative sampler excludes full observed basket; diagnose full-catalogue checkout rather than dropping baskets')
            prices=np.pad(np.exp(tr['dlp']),((0,0),(1,0)),constant_values=1).astype('float32');avail=np.ones_like(prices)
            rng=np.random.default_rng(seed);best=-np.inf;history=[];stale=0;significant=-np.inf;start=time.monotonic();stop='epoch_limit'
            for epoch in range(21):
                losses=[]
                if epoch:
                    targets,prefix,future,rows=epoch_rows(tr,rng)
                    for k in range(0,len(rows),128):
                        if time.monotonic()-start>=180:break
                        sl=slice(k,k+128);ii=rows[sl]
                        value=float(model.train_step(targets[sl],prefix[sl],future[sl],tr['house'][ii].astype('int32'),np.zeros(len(ii),'int32'),prices[ii],avail[ii],ii.astype('int32')))
                        assert np.isfinite(value);losses.append(value)
                w=weights(model);vl=float(sh.set_ll(sh.base_utility(w,va),sh.interaction_matrix(w),va['member_flat']).mean())
                if vl>best:best=vl;np.savez_compressed(path/'best_weights.npz',**w)
                if vl>significant+.001:significant=vl;stale=0
                else:stale+=1
                row={'epoch':epoch,'native_loss':float(np.mean(losses)) if losses else None,'validation_ll':vl,'seconds':time.monotonic()-start};history.append(row);sh.write(path/'history.json',history)
                print(json.dumps({'rank':rank,'seed':seed,**row}),flush=True)
                if time.monotonic()-start>=180:stop='time_limit';break
                if epoch>=5 and stale>=4:stop='validation_patience';break
            row={'rank':rank,'seed':seed,'best_validation_ll':best,'seconds':time.monotonic()-start,'stop':stop,'epochs':len(history)-1,'convergence_certified':False,'checkpoint':str(path/'best_weights.npz'),'label':'Pinned Choice-Learn myopic SHOPPER + shared observables; native sampled loss/Adam, not original Bayesian/thinking-ahead fit.'}
            sh.write(path/'fit.json',row);summaries.append(row);sh.write(OUT/'native_fit_records.json',summaries)
    selected=max(summaries,key=lambda r:r['best_validation_ll']);sh.write(OUT/'native_selected.json',selected)
    print(json.dumps({'phase':'native_complete','selected':selected}),flush=True)
if __name__=='__main__':main()
