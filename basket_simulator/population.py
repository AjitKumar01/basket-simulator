"""A finite empirical shopper mixture for explicitly unknown shoppers.

Mix basket probabilities, not utility coefficients. The prior is built only from
training household IDs/frequencies, conditional on a retained nonempty trip.
It does not infer a new shopper's preferences or identify visit propensity.
"""
import numpy as np
from scipy.special import logsumexp


class PopulationBasketSimulator:
    def __init__(self,model,households,weights):
        self.model=model;self.C,self.P,self.R=model.C,model.P,model.R
        raw=np.asarray(households);weights=np.asarray(weights,dtype=float)
        if raw.ndim!=1 or not len(raw) or weights.shape!=raw.shape:
            raise ValueError('Nonempty matching household/weight vectors required')
        if not np.isfinite(raw).all() or not np.equal(raw,raw.astype(int)).all():
            raise ValueError('Household IDs must be finite integers')
        ids=raw.astype(int)
        limit=int(model.config['users']) if model.config.get('model_family')=='product_interactions' else len(model.s['theta'])
        if (ids<=0).any() or (ids>=limit).any() or len(np.unique(ids))!=len(ids):
            raise ValueError('Prior requires unique known household IDs, excluding row0')
        if not np.isfinite(weights).all() or (weights<=0).any():raise ValueError('Positive finite weights required')
        weights=weights/weights.max()
        self.households=ids.copy();self.weights=weights/weights.sum()

    @classmethod
    def from_training(cls,model,household_ids,representatives=128,seed=0):
        if not isinstance(representatives,int) or representatives<1:raise ValueError('Positive representatives required')
        values=np.asarray(household_ids)
        if values.ndim!=1 or not np.isfinite(values).all() or not np.equal(values,values.astype(int)).all():
            raise ValueError('Training household IDs must be a finite integer vector')
        values=values.astype(int)
        if (values<0).any():raise ValueError('Household IDs must be nonnegative')
        values=values[values>0]
        if not len(values):raise ValueError('No known training households')
        ids,counts=np.unique(values,return_counts=True)
        draws=np.random.default_rng(seed).choice(ids,representatives,p=counts/counts.sum())
        ids,counts=np.unique(draws,return_counts=True)
        return cls(model,ids,counts)

    def _context(self,context,h):
        if context.get('house')!=0:raise ValueError('Population prior is only for explicit unknown household0')
        return {**context,'house':int(h)}

    def _combine(self,rows):
        incidence=np.sum([w*r['incidence'] for w,r in zip(self.weights,rows)],axis=0)
        presence=np.sum([w*r['category_presence'] for w,r in zip(self.weights,rows)],axis=0)
        result={'incidence':incidence,'category_presence':presence,'expected_distinct_size':float(incidence.sum()),
                'integration':'finite shopper mixture; '+rows[0]['integration'],'shopper_components':len(rows)}
        for key in ('category_standard_error','incidence_standard_error'):
            if key in rows[0]:result[key]=np.sqrt(np.sum([w*w*r[key]**2 for w,r in zip(self.weights,rows)],axis=0))
        return result

    def predict(self,context,draws=8192,seed=0,method='auto'):
        rows=[self.model.predict(self._context(context,h),draws,seed+i,method) for i,h in enumerate(self.households)]
        return self._combine(rows)

    def log_probability(self,context,basket):
        scores=[self.model.log_probability(self._context(context,h),basket) for h in self.households]
        return float(logsumexp(np.asarray(scores)+np.log(self.weights)))

    def generate(self,context,draws=1,seed=0):
        if not isinstance(draws,int) or draws<1:raise ValueError('Positive draws required')
        rng=np.random.default_rng(seed);component=rng.choice(len(self.households),draws,p=self.weights)
        result=np.empty((draws,self.C,self.P),dtype=bool)
        for i in np.unique(component):
            rows=np.flatnonzero(component==i)
            result[rows]=self.model.generate(self._context(context,self.households[i]),len(rows),int(rng.integers(0,2**31)))
        return result

    def what_if(self,context,product,factor,draws=8192,seed=0):
        rows=[self.model.what_if(self._context(context,h),product,factor,draws,seed+i) for i,h in enumerate(self.households)]
        before=self._combine([r['before'] for r in rows]);after=self._combine([r['after'] for r in rows])
        result={'before':before,'after':after,'incidence_change':after['incidence']-before['incidence'],
                'size_change':after['expected_distinct_size']-before['expected_distinct_size']}
        for key in ('incidence_change_standard_error','size_change_standard_error'):
            if key in rows[0]:result[key]=np.sqrt(np.sum([w*w*r[key]**2 for w,r in zip(self.weights,rows)],axis=0))
        return result

    def recommend(self,context,revealed=None,draws=8192,seed=0):
        if revealed is None:
            scores=self.predict(context,draws,seed)['incidence'].ravel()
            return np.argsort(-scores,kind='stable'),scores
        y=np.asarray(revealed)
        if y.shape!=(self.C,self.P) or not np.isin(y,(0,1)).all():raise ValueError('Binary revealed basket required')
        scores=np.full(self.C*self.P,-np.inf)
        for j in np.flatnonzero(~y.astype(bool).ravel()):
            candidate=y.copy();candidate.flat[j]=1;scores[j]=self.log_probability(context,candidate)
        ids=np.flatnonzero(np.isfinite(scores))
        return ids[np.argsort(-scores[ids],kind='stable')],scores
