"""Rebuild the studied ERIM projection using training-only catalogue and mappings."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .. import common as c


def main():
    source = Path(os.environ['BASKET_DATA_SOURCE'])
    data = c.OUT/'data'
    data.mkdir(exist_ok=False)
    inputs = [source/'transactions.parquet',source/'store_week_prices.parquet']
    c.write('preparation_manifest.json',{str(p):c.digest(p) for p in inputs})
    tx = pd.read_parquet(inputs[0])
    panel = pd.read_parquet(inputs[1])
    train = tx[tx.period<=30]
    # Both category eligibility and references are restricted to training periods.
    retail = panel[(panel.price_source=='retail_aggregate')&(panel.average_price>0)]
    categories = sorted(set(train.category)&set(retail[retail.period<=30].category))
    frequency = train.groupby(['category','product_id']).size().rename('training_lines').reset_index()
    catalogue = frequency[frequency.category.isin(categories)].sort_values(
        ['category','training_lines','product_id'],ascending=[True,False,True]).groupby('category').head(4)
    catalogue = catalogue.sort_values(['category','product_id']).reset_index(drop=True)
    if len(catalogue)!=len(categories)*4 or len(categories)<3:
        raise ValueError('Require at least three categories, each with four training products')
    catalogue.to_csv(data/'products.csv',index=False)
    products = catalogue.product_id.tolist()
    C,P,J = len(categories),4,len(products)
    item = {p:i for i,p in enumerate(products)}
    selected = tx[tx.product_id.isin(item)].copy()
    training = selected[selected.period<=30]
    users = {h:i+1 for i,h in enumerate(sorted(training.customer_id.unique()))}
    stores = {s:i+1 for i,s in enumerate(sorted(training.store_id.unique()))}
    retail = retail[retail.product_id.isin(item)]
    if retail.duplicated(['product_id','store_id','period']).any():
        raise ValueError('Duplicate retail price rows')
    ref = retail[retail.period<=30].groupby('product_id').average_price.median().reindex(products).to_numpy()
    if not np.isfinite(ref).all() or not (ref>0).all():
        raise ValueError('Training reference price missing')
    S,T = int(tx.store_id.max())+1,int(tx.period.max())+1
    price = np.broadcast_to(ref,(S,T,J)).copy()
    display = np.zeros_like(price);mail = np.zeros_like(price)
    source_period = np.zeros(price.shape,dtype=int)
    for (p,s),group in retail.groupby(['product_id','store_id']):
        s,j = int(s),item[p]
        group = group.sort_values('period')
        for week in range(1,T):
            rows = group[group.period<=week]
            if len(rows):
                row = rows.iloc[-1]
                price[s,week,j] = row.average_price
                source_period[s,week,j] = int(row.period)
                display[s,week,j] = any(str(row[k])=='Y' for k in ('end_display','front_display','aisle_display','other_display'))
                mail[s,week,j] = str(row.advertised)=='Y'
    audit = {'dataset':'erim','catalogue_products':J,'categories':C,
             'price_contract':'Same-period retrospective retail sales average, else earlier row, else training median. Reproduces the studied predictive contract; not pre-decision shelf offers.',
             'availability':'All selected products assumed offered','splits':{}}
    for name,lo,hi in [('train',1,30),('validation',31,40),('test',41,51)]:
        full = tx[tx.period.between(lo,hi)]
        part = selected[selected.period.between(lo,hi)]
        if (part.groupby('basket_id').customer_id.nunique()>1).any() or (part.groupby('basket_id').store_id.nunique()>1).any() or (part.groupby('basket_id').period.nunique()>1).any():
            raise ValueError('Basket metadata not unique')
        metadata = part.groupby('basket_id').agg(house=('customer_id','first'),store=('store_id','first'),week=('period','first')).sort_index()
        rows = {b:i for i,b in enumerate(metadata.index)}
        y = np.zeros((len(metadata),J),dtype=np.uint8)
        y[part.basket_id.map(rows),part.product_id.map(item)] = 1
        s,w = metadata.store.to_numpy(int),metadata.week.to_numpy(int)
        np.savez_compressed(data/f'{name}.npz',
            house=metadata.house.map(users).fillna(0).to_numpy(np.int64),
            store=metadata.store.map(stores).fillna(0).to_numpy(np.int64),week=w,
            dlp=np.log(price[s,w]/ref).reshape(-1,C,P),disp=display[s,w].reshape(-1,C,P),
            mail=mail[s,w].reshape(-1,C,P),member=y.reshape(-1,C,P),
            raw_household=metadata.house.to_numpy(),raw_store=s,basket_id=metadata.index.to_numpy(),
            price_source_period=source_period[s,w].reshape(-1,C,P))
        audit['splits'][name] = {'baskets_projected':len(y),'baskets_original':int(full.basket_id.nunique()),
            'line_retained_fraction':float(full.product_id.isin(item).mean()),
            'current_price_fraction':float((source_period[s,w]==w[:,None]).mean()),
            'training_reference_fraction':float((source_period[s,w]==0).mean())}
    np.save(data/'reference_price.npy',ref)
    c.write('data/config.json',{'C':C,'P':P,'R':P,'users':len(users)+1,'stores':len(stores)+1,
        'calendar_support':[1,30],'categories':categories,'products':products,
        'preference_mode':'direct','learn_inclusive_price':True,'chain_rank':0,'lbfgs_history':3,
        'memory_budget_mib':14000,'scope':'Nonempty ERIM distinct-product projections; retrospective observational price proxies'})
    c.write('data/id_maps.json',{'household':{str(k):v for k,v in users.items()},'store':{str(k):v for k,v in stores.items()}})
    c.write('data_audit.json',audit)
    c.write('prepared_manifest.json',{str(p):c.digest(p) for p in data.iterdir()})
    c.status('prepared',dataset='erim',products=J)
