"""Training-only catalogue; compressed basket labels and shared lagged offer contexts."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from ..common import OUT, ROOT
SRC=Path(os.environ['BASKET_DATA_SOURCE']).resolve()
BASE=SRC/'data' if (SRC/'data/tx.parquet').exists() else SRC
RAW=Path(os.environ.get('BASKET_RAW_SOURCE',SRC/'dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV')).resolve()

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert not (OUT/'data/config.json').exists();(OUT/'data').mkdir(exist_ok=True)
 paths=[BASE/'tx.parquet',BASE/'price_store_week.parquet',BASE/'price_week.parquet',RAW/'product.csv',RAW/'causal_data.csv',OUT/'protocol.json',Path(__file__)]
 write('preparation_manifest.json',{str(p):digest(p) for p in paths})
 tx=pd.read_parquet(paths[0],columns=['household_key','BASKET_ID','WEEK_NO','PRODUCT_ID','STORE_ID','QUANTITY'])
 tx=tx[tx.WEEK_NO.between(9,101)];tr=tx[tx.WEEK_NO<=82]
 prod=pd.read_csv(RAW/'product.csv').merge(tr.groupby('PRODUCT_ID').size().rename('training_lines'),on='PRODUCT_ID')
 eligible=prod.groupby('COMMODITY_DESC').filter(lambda a:len(a)>=16)
 scores=eligible.groupby('COMMODITY_DESC').training_lines.sum().reset_index().sort_values(['training_lines','COMMODITY_DESC'],ascending=[False,True]);cats=sorted(scores.COMMODITY_DESC.head(64).tolist())
 prod=eligible[eligible.COMMODITY_DESC.isin(cats)].sort_values(['COMMODITY_DESC','training_lines','PRODUCT_ID'],ascending=[True,False,True]).groupby('COMMODITY_DESC').head(16).copy();assert len(prod)==1024
 prod=prod.sort_values(['COMMODITY_DESC','training_lines','PRODUCT_ID'],ascending=[True,False,True]);prod['model_item']=np.arange(1024);prod.to_csv(OUT/'data/products.csv',index=False)
 pm=dict(zip(prod.PRODUCT_ID,prod.model_item));hm={int(h):i+1 for i,h in enumerate(sorted(tr.household_key.unique()))};sm={int(s):i+1 for i,s in enumerate(sorted(tr.STORE_ID.unique()))}
 selected=tx[tx.PRODUCT_ID.isin(pm)].copy();selected['item']=selected.PRODUCT_ID.map(pm)
 # Offer rows are indexed by actual store/week, independent of basket outcomes.
 contexts=selected[['STORE_ID','WEEK_NO']].drop_duplicates().sort_values(['STORE_ID','WEEK_NO']).reset_index(drop=True);cm={(int(s),int(w)):i for i,(s,w) in enumerate(contexts.itertuples(index=False,name=None))};K=len(contexts)
 store_ids=sorted(contexts.STORE_ID.unique());sgrid={int(s):i for i,s in enumerate(store_ids)};S=len(store_ids);W=103;J=1024
 national=pd.read_parquet(BASE/'price_week.parquet');national=national[national.PRODUCT_ID.isin(pm)&national.price.gt(0)&national.WEEK_NO.between(1,101)]
 reference=national[national.WEEK_NO.between(9,82)].groupby('PRODUCT_ID').price.median().reindex(prod.PRODUCT_ID).to_numpy();assert np.isfinite(reference).all() and (reference>0).all()
 nprice=np.full((W,J),np.nan,np.float32)
 nprice[national.WEEK_NO.to_numpy(),national.PRODUCT_ID.map(pm).to_numpy()]=national.price.to_numpy()
 natlast=np.full(J,np.nan,np.float32);natlag=np.empty_like(nprice)
 for w in range(W):natlag[w]=natlast;natlast=np.where(np.isfinite(nprice[w]),nprice[w],natlast)
 prices=pd.read_parquet(BASE/'price_store_week.parquet');prices=prices[prices.PRODUCT_ID.isin(pm)&prices.STORE_ID.isin(sgrid)&prices.WEEK_NO.between(1,101)&prices.price.gt(0)]
 grid=np.full((S,W,J),np.nan,np.float32);grid[prices.STORE_ID.map(sgrid).to_numpy(),prices.WEEK_NO.to_numpy(),prices.PRODUCT_ID.map(pm).to_numpy()]=prices.price.to_numpy()
 dlp=np.empty((K,64,16),np.float32);source=np.empty((K,1024),np.uint8);age=np.empty((K,1024),np.int16)
 for s in store_ids:
  last=np.full(J,np.nan,np.float32);lastweek=np.zeros(J,int)
  for w in range(W):
   key=(int(s),w)
   if key in cm:
    i=cm[key];offers=np.where(np.isfinite(last),last,np.where(np.isfinite(natlag[w]),natlag[w],reference));dlp[i]=np.log(offers/reference).reshape(64,16)
    source[i]=np.where(np.isfinite(last),0,np.where(np.isfinite(natlag[w]),1,2));age[i]=np.where(np.isfinite(last),w-lastweek,-1)
   current=grid[sgrid[int(s)],w];available=np.isfinite(current);last=np.where(available,current,last);lastweek=np.where(available,w,lastweek)
 del grid,prices,national,nprice,natlag
 display=np.zeros((K,1024),np.uint8);mail=display.copy();coverage=[10000,-1];nraw=0;nselected=0;unpromoted=0
 for ch in pd.read_csv(RAW/'causal_data.csv',dtype={'PRODUCT_ID':np.int64,'STORE_ID':np.int64,'WEEK_NO':np.int16,'display':str,'mailer':str},chunksize=1000000):
  coverage=[min(coverage[0],int(ch.WEEK_NO.min())),max(coverage[1],int(ch.WEEK_NO.max()))];nraw+=len(ch)
  ch=ch[ch.PRODUCT_ID.isin(pm)&ch.STORE_ID.isin(sgrid)&ch.WEEK_NO.between(9,101)]
  row=np.array([cm.get((int(s),int(w)),-1) for s,w in zip(ch.STORE_ID,ch.WEEK_NO)],dtype=np.int64);valid=row>=0;ch=ch[valid];row=row[valid];item=ch.PRODUCT_ID.map(pm).to_numpy(dtype=np.int64);d=(ch.display.fillna('0')!='0').to_numpy(np.uint8);m=(ch.mailer.fillna('0')!='0').to_numpy(np.uint8)
  np.maximum.at(display,(row,item),d);np.maximum.at(mail,(row,item),m);nselected+=len(ch);unpromoted+=int(((d==0)&(m==0)).sum())
 assert coverage==[9,101] and unpromoted==0
 for name,a in [('dlp',dlp),('disp',display.reshape(K,64,16)),('mail',mail.reshape(K,64,16))]:np.save(OUT/'data'/f'{name}.npy',a)
 np.save(OUT/'data/reference_price.npy',reference);contexts.to_csv(OUT/'data/contexts.csv',index=False)
 result={'catalogue_products':1024,'categories':64,'household_ids_training':len(hm),'store_ids_training':len(sm),'context_rows':K,'promotion_file':{'coverage':coverage,'raw_rows':nraw,'selected_context_rows':nselected,'unpromoted_records':unpromoted},'splits':{}}
 for name,lo,hi in [('train',9,82),('validation',83,90),('test',91,101)]:
  full=tx[tx.WEEK_NO.between(lo,hi)];part=selected[selected.WEEK_NO.between(lo,hi)].copy();group=['BASKET_ID','STORE_ID'];metadata=part.groupby(group).agg(household_key=('household_key','first'),WEEK_NO=('WEEK_NO','first')).reset_index();assert (part.groupby(group).household_key.nunique()==1).all() and (part.groupby(group).WEEK_NO.nunique()==1).all()
  bm={(int(b),int(s)):i for i,(b,s) in enumerate(metadata[group].itertuples(index=False,name=None))};part=part.drop_duplicates(group+['item']);rows=np.array([bm[(int(b),int(s))] for b,s in zip(part.BASKET_ID,part.STORE_ID)]);y=np.zeros((len(metadata),1024),np.uint8);y[rows,part.item.to_numpy()]=1;ci=np.array([cm[(int(s),int(w))] for s,w in zip(metadata.STORE_ID,metadata.WEEK_NO)])
  np.savez_compressed(OUT/'data'/f'{name}.npz',house=metadata.household_key.map(hm).fillna(0).to_numpy(np.int64),store=metadata.STORE_ID.map(sm).fillna(0).to_numpy(np.int64),week=metadata.WEEK_NO.to_numpy(np.int64),context=ci,packed_member=np.packbits(y,axis=1),raw_household=metadata.household_key.to_numpy(),raw_store=metadata.STORE_ID.to_numpy(),basket_id=metadata.BASKET_ID.to_numpy())
  result['splits'][name]={'baskets_original':int(full[group].drop_duplicates().shape[0]),'baskets_projected':len(y),'distinct_original':int(full.drop_duplicates(group+['PRODUCT_ID']).shape[0]),'distinct_projected':len(part),'line_retained_fraction':float(full.PRODUCT_ID.isin(pm).mean()),'size_mean':float(y.sum(-1).mean()),'size_max':int(y.sum(-1).max()),'category_size_max':int(y.reshape(-1,64,16).sum(-1).max()),'households':int(metadata.household_key.nunique()),'unknown_household_fraction':float((~metadata.household_key.isin(hm)).mean()),'unknown_store_fraction':float((~metadata.STORE_ID.isin(sm)).mean()),'price_source_fraction':{label:float((source[ci]==i).mean()) for i,label in enumerate(['prior_store','prior_national','training_reference'])},'prior_store_age_median':float(np.median(age[ci][age[ci]>=0]))}
 config={'C':64,'P':16,'R':16,'users':len(hm)+1,'stores':len(sm)+1,'calendar_support':[9,82],'products':prod.PRODUCT_ID.astype(str).tolist(),'categories':cats,'preference_mode':'direct','learn_inclusive_price':True,'chain_rank':4,'lbfgs_history':3,'memory_budget_mib':14000,'scope':'nonempty distinct1024-product projections; lagged price proxies; fully offered assumption'}
 write('data/config.json',config);write('data/id_maps.json',{'household':hm,'store':sm});write('data_audit.json',result)
 write('prepared_manifest.json',{str(p):digest(p) for p in (OUT/'data').iterdir() if p.is_file()});print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
