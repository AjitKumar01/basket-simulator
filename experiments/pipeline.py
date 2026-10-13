"""Durable full-data fits. A failed prerequisite stops dependent fits."""
import copy,fcntl,json,os,resource,sys,time,traceback
from pathlib import Path
from . import common as c
import numpy as np,torch

def selected(m,name,row):
 c.save_model(m,c.OUT/'selected'/name);ledger=json.loads((c.OUT/'fit_records.json').read_text()) if (c.OUT/'fit_records.json').exists() else {};ledger[name]=row;c.write('fit_records.json',ledger)
def full_score(m,data):
 with torch.no_grad():return float(torch.cat([m.log_probability(*data.batch(slice(i,i+512))) for i in range(0,len(data),512)]).mean())
def gated(row,name):
 if not row['final_stationarity_passed']:raise RuntimeError('Unresolved final gradient failure in '+name)
def create_structure(parent,blocks):
 cfg={**parent.config,'independent_only':False,'trainable_parameters':blocks};m=c.CachedStructure(cfg);m.load_state_dict(parent.state_dict());return m

def size_cache(model,data,name):
 dest=c.OUT/'cache'/name;dest.mkdir(parents=True,exist_ok=True);path=dest/'within.npy';a=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=(len(data),))
 for i in range(0,len(data),256):
  ctx,y=data.batch(slice(i,i+256))
  with torch.no_grad():u,_=model.terms(ctx);n=y.sum((1,2)).long();e=c.log_esp(u,int(n.max()));a[i:i+256]=((u*y.reshape(-1,model.J)).sum(-1)-e.gather(1,n[:,None]).ravel()).numpy()
  if i%8192==0:c.status('size_cache',rows=i,total=len(data))
 a.flush();return {**data.ctx,'_size_within':c.Shared(np.arange(len(data)),np.load(path,mmap_mode='r'))},data.y

def run():
 start=time.monotonic();cfg=json.loads((c.OUT/'data/config.json').read_text());tr,va=c.Data('train'),c.Data('validation')
 # Freeze scientific source/data identity before fitting; validate at resume.
 from .provenance import freeze
 freeze(c.OUT, 'fit_manifest.json', include_data=True)
 assert json.loads((c.OUT/'preflight.json').read_text())['state']=='passed'
 pooled_start=time.monotonic();pooled=c.independent_start(cfg,tr.y,c.SEED);pooled_seconds=time.monotonic()-pooled_start;p=c.Independent(pooled.config);p.load_state_dict(pooled.state_dict());selected(p,'popularity',{'label':'Training-only pooled conditional independent MLE','final_stationarity_passed':True,'validation_ll':full_score(p,va),'seconds':pooled_seconds})
 # All data used; same direct tastes and validation-only ridge selection.
 preferences=[]
 for name in ('no_price','independent'):
  candidates=[]
  for ridge in (1.,10.):
   m=c.Independent(p.config);m.load_state_dict(p.state_dict())
   if name=='no_price':
    m.config['trainable_parameters']=[n for n,q in m.named_parameters() if q.requires_grad and n!='choice_price'];m.choice_price.requires_grad_(False)
   m,row=c.run_fit(m,f'{name}_ridge{ridge:g}',tr.pair(),va.pair(),ridge);candidates.append((m,row))
  m,row=max(candidates,key=lambda x:x[1]['best_validation_ll']);gated(row,name);selected(m,name,{'fits':[r for _,r in candidates],'selected':row,'validation_ll':row['best_validation_ll'],'final_stationarity_passed':row['final_stationarity_passed']})
  if name=='independent':parent=m
 # Shared frozen ESP cache, exact mathematical equivalence checked preflight.
 category_blocks=['linear_p','linear_q'] if cfg['C']>10 else ['V']
 stage=create_structure(parent,['rho','kappa']+category_blocks);tcache=c.materialize_cache(stage,tr,'structure_train','structure');vcache=c.materialize_cache(stage,va,'structure_validation','structure')
 for name,blocks in [('independent_categories',['rho','kappa']),('basic',['rho','kappa']+category_blocks)]:
  m=create_structure(parent,blocks)
  warm=[]
  for blockset,label in [(['rho'],'count_warmup'),([n for n in blocks if n!='rho'],'category_warmup')]:
   w=create_structure(parent,blockset);w.load_state_dict(m.state_dict());w,row=c.run_fit(w,name+'_'+label,tcache,vcache);gated(row,name+'_'+label);w.load_state_dict(torch.load(c.OUT/'fits'/(name+'_'+label)/('continuation/final.pt' if 'initial_attempt' in row else 'final.pt'),weights_only=True));m.load_state_dict(w.state_dict());warm.append(row)
  m,row=c.run_fit(m,name,tcache,vcache);gated(row,name);parentval=full_score(parent,va)
  if row['best_validation_ll']<parentval:
   m=create_structure(parent,blocks);row['parent_retained']=True;row['effective_validation_ll']=parentval
  else:row['effective_validation_ll']=row['best_validation_ll']
  selected(m,name,{'fit':row,'warmups':warm,'validation_ll':row['effective_validation_ll'],'final_stationarity_passed':True})
  if name=='basic':basic=m
 del tcache,vcache
 # Projected independent reference is explicit in the released product head.
 head=c.ProductInteractionModel(basic,4,c.SEED);tc=c.materialize_cache(head,tr,'product_train','product');vc=c.materialize_cache(head,va,'product_validation','product');head,row=c.run_fit(head,'product',tc,vc)
 parentval=full_score(basic,va);row['original_chain_parent_validation_ll']=parentval;row['accepted']=row['best_validation_ll']>parentval+1e-10
 if row['accepted']:gated(row,'product');selected(head,'product',{'fit':row,'validation_ll':row['best_validation_ll'],'final_stationarity_passed':True})
 else:selected(basic,'product_fallback',{'fit':row,'validation_ll':parentval,'label':'Optional head rejected; converged basic parent retained','final_stationarity_passed':True})
 del tc,vc,head
 for name in ('mission','neural','size_aware'):
  if time.monotonic()-start>14400:c.write('budget_stop.json',{'next_model':name,'elapsed_seconds':time.monotonic()-start});break
  m=c.MissionESP(parent,c.SEED) if name=='mission' else c.Neural(parent,c.SEED) if name=='neural' else c.SizeESP(parent,tr.y)
  tp,vp=tr.pair(),va.pair()
  if name=='size_aware':tp,vp=size_cache(m,tr,'size_train'),size_cache(m,va,'size_validation')
  m,row=c.run_fit(m,name,tp,vp)
  selected(m,name,{'fit':row,'validation_ll':row['best_validation_ll'],'final_stationarity_passed':row['final_stationarity_passed']})
 import platform
 c.write('fits_complete.json',{'state':'complete' if not (c.OUT/'budget_stop.json').exists() else 'budget_stop','seconds':time.monotonic()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if platform.system()=='Darwin' else 1024)});c.status('fits_complete',seconds=time.monotonic()-start)
 # Launch scoring only after fitting and its gating have completed.
 from . import evaluate
 evaluate.main()
